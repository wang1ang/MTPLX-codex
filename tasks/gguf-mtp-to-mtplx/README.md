# GGUF MTP head → MTPLX sidecar

Extract the MTP/nextn head from a **GGUF that ships an MTP head** (e.g. the
`*-MTP-*.gguf` files used by llama.cpp `--spec-type draft-mtp`), turn it into an
MTPLX-compatible `mtp.safetensors` sidecar, then pair it with the **full-precision
safetensors base model** and run Forge.

> This is NOT a one-shot script. Architecture / quantization / norm encoding /
> sharding differ per model, so **judge before you act each time**. What's here
> is a set of tools plus a decision checklist, not a fixed pipeline.

## Why this is needed
Some HF repos publish a BF16 safetensors body that **dropped the MTP head** during
fine-tuning, while only the companion GGUF repo carries a "restored" MTP head.
MTPLX runs MLX, Forge **does not train** an MTP head (it only carries over an
existing one), and Forge **rejects GGUF sources**. So the GGUF MTP head must be
hand-converted into an MLX safetensors sidecar.

## Tools
- `gguf_peek.py <file>` — pure-stdlib GGUF header parser; lists every tensor's
  name/dtype/shape/offset plus architecture metadata. **Use it first** to see
  what the MTP tensors are named and which block they live in.
- `gguf_plan.py <file> > plan.json` — computes the **absolute byte ranges** of
  the target block's tensors and how far to download (`download_through`).
  Defaults to extracting `blk.<last layer>.*`.
- `extract_mtp.py <file> <plan.json> <out/mtp.safetensors>` — extracts per the
  mapping and writes an MLX safetensors. The GGUF→MTPLX key map, orientation,
  and dtype rules are documented at the top of the script.

## Checklist — decide these every time

1. **Does MTPLX recognize the base architecture?**
   Check `ArchitectureSupport` in `mtplx/backends/registry.py`. The Qwen3.5
   family (`qwen3_5`, `qwen3_6`) maps to the `qwen3-next-mtp` backend (`qwen3_5`
   is in its aliases). If it's not in the table, stop and reconsider.

2. **Which GGUF block holds the MTP head, and what are the tensor names?**
   Inspect with `gguf_peek.py`. In the Qwen3.5 family the MTP head is the last
   block (`block_count-1`), with prefix `blk.<N>.nextn.*` plus standard attn/ffn.
   `nextn_predict_layers=1` means a single MTP layer.

3. **Is the GGUF→MTPLX key mapping correct?**
   MTPLX's expected dense key set is `EXPECTED_MTP_KEYS` in
   `mtplx/constants.py` (15 keys for dense; MoE has `EXPECTED_QWEN_MOE_*`).
   `extract_mtp.py`'s `SUFFIX_MAP` matches dense. **MoE models need a different
   mapping** (experts / shared_expert / switch_mlp).

4. **Do you need to dequantize?**
   Using `*-MTP-BF16.gguf` you **don't** (tensors are BF16/F32, read bytes
   directly). If only Q4_K_M etc. exist, you must add k-quant super-block
   decoding — painful, so prefer the BF16 GGUF.

5. **Do linear weights need transposing?**
   GGUF dims are reversed vs row-major; after reversing, 2-D weights are already
   HF `[out, in]`, so **no transpose**. After extraction, sanity-check shapes
   against the base model `config.json` (hidden / heads×head_dim / intermediate).
   Note Qwen3.5 has `attn_output_gate`, so q_proj output dim is `2*hidden`.

6. **Are the norms delta-encoded (weight-1)?** ← easiest trap
   After extraction, look at norm means:
   - **Mean ~1 (e.g. 0.5–2.5)** → already real values, **do NOT** set
     `mtplx_mtp_norm_encoding` in config.
   - **Mean ~0 (with negatives)** → delta-encoded; set
     `"mtplx_mtp_norm_encoding": "delta"` in the assembled `config.json` so MTPLX
     adds +1 at load (see `_restore_delta_encoded_mtp_norms` in
     `mtplx/mtp_patch.py`). Pick exactly one, or you get an off-by-one.
   `extract_mtp.py` prints a WARNING if this looks wrong.
   (Measured for empero-ai Qwythos-9B's GGUF: norms are real values — do NOT
   set delta.)

## Assemble + Forge (only after the checklist)
```bash
SRC=/path/to/<model>-src              # full-precision safetensors base dir
cp mtp.safetensors "$SRC/"
# In $SRC/config.json add: mlx_lm_extra_tensors.mtp_file = "mtp.safetensors"
#   (for delta-encoded models also add mtplx_mtp_norm_encoding="delta")

cd /path/to/MTPLX
python3 -m mtplx.cli forge probe "$SRC" --json     # expect verdict: forgeable
python3 -m mtplx.cli forge build \
  --repo "$SRC" --out ~/models/forge-runs --run-id run-$(date +%s) \
  --recipe '{"body_bits":4,"body_group_size":64,"body_mode":"affine","mtp_policy":"keep_bf16"}' \
  --branded-name "<Name>-MTPLX-Speed" --json
```

## Quantization notes
- **Don't** reuse the GGUF's quant coefficients: k-quant and MLX affine are
  incompatible formats, and the coefficients are recomputed deterministically
  from BF16 anyway. Quantize straight from the BF16 body.
- Pick `body_bits` 4/6/8 to trade quality vs size; **keep the MTP head bf16**
  (`mtp_policy=keep_bf16`) — it's small but directly drives acceptance.
- Correctness after quantization isn't guesswork: `forge build` ends with a
  **contract calibration + exact rejection-sampling verification** that reports
  acceptance and tok/s speedup, and refuses to ship if it doesn't pass.

## Download tips
- The base model is often a single large `model.safetensors` (no index.json).
  `hf download` is single-connection and slow; `aria2c -x16 -s16` is much
  faster. The tail-end hole-filling slows down — that's normal.
- Extracting the MTP head only needs the first ~500MB of the GGUF (the MTP block
  is at the front). `gguf_plan.py` reports `download_through`. Note HF range
  requests sometimes truncate around ~493MB, so request extra margin or resume
  with `-C -`.
```
