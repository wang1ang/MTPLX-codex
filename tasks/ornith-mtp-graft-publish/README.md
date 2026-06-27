# Ornith-1.0-35B: graft MTP → 4-bit → fix template → publish

End-to-end record of turning **deepreinforce-ai/Ornith-1.0-35B** (qwen35moe, 35B-A3B,
ships **without** MTP) into a published, working MTPLX model with native MTP speculative
decoding: **[wang-yang/Ornith-1.0-35B-MTPLX](https://huggingface.co/wang-yang/Ornith-1.0-35B-MTPLX)**.

> Companion task `gguf-mtp-to-mtplx/` covers extracting an MTP head from a **GGUF**.
> This task instead grafts from a **safetensors** MTP source, then quantizes, fixes the
> chat template for agents, and publishes. Judge each step; this is a record, not a fixed script.

## Result

| Mode | tok/s | Speedup | Acceptance |
|---|---|---|---|
| AR | 76.3 | 1.00× | — |
| MTP D1 | 103.6 | 1.36× | 89.6% |
| MTP D3 (winner) | **116.4** | **1.53×** | 91.5 / 80.3 / 65.6% |

verdict `mtp_depth_wins`, inspect tier `verified`. Body 4-bit affine g64, MTP 4-bit affine g64.

## The pipeline (and why each step)

### 1. Get the MTP source — Qwen3.5-35B-A3B official
Ornith's upstream base is **Qwen/Qwen3.5-35B-A3B** (Apache-2.0), which ships the official
1-layer MoE-MTP. Dimensions match Ornith exactly (hidden 2048 / 40 layers / 256 experts /
vocab 248320). Only shards **13+14** (~7G) carry the 785 `mtp.*` tensors — download just those,
not the full repo. Source is already safetensors with `mtp.*` keys → no GGUF extraction needed,
just `mx.load` and slice.

### 2. Graft + Forge
Forge has **no native grafting** — it does `mlx_lm convert` + contract calibration + verify,
and requires the MTP sidecar already placed. Assemble = copy `mtp.safetensors` into the base
dir + set `mlx_lm_extra_tensors.mtp_file` in config.json. Then `mtplx forge`.

### 3. The 4-bit MTP decision — THIS is what makes it fast
A **bf16** graft of this MTP was a net **~20× SLOWDOWN** despite 92% acceptance. Cause: the MTP
is a full **256-expert MoE layer**; at bf16 its draft forward costs **67–143 ms/token** (doesn't
hit the fast MoE kernel). The bottleneck is **draft cost, not acceptance, not verify.**
Quantizing the MTP experts to **4-bit affine** drops draft cost to **~2.5 ms/token (~27–51×)**
with negligible acceptance loss → flips to a real 1.53× speedup.
Done by adding `mtplx_mtp_quantization:{bits:4,group_size:64,mode:affine}` to config.json
(MTPLX `nn.quantize`s the MTP MoE at load via `mtp_patch._quantize_mtp_module`, policy "all");
forge `mtp_policy=requantize` + `--allow-degraded-mtp`.

### 4. Fix the chat template — or agents loop forever
Ornith's template has `raise_exception('System message must be at the beginning.')`. Codex (and
any agent) puts system not-first → it raises → tool result never returns → **agent re-sends the
same command forever**. Fix: change that one line (non-first system → render normally as an
`<|im_start|>system` block) — same edit as the Qwythos fix. Keep all other guards + `<think>` +
tool_call logic. **Write the fixed template to BOTH** `chat_template.jinja` AND
`tokenizer_config.json.chat_template` (the latter was empty; transformers/vLLM read it first).
Verify rendering 4 ways: system-first OK, system-not-first OK (no raise), tool_call markers still
emit, `<think>` preserved.

### 5. Publish
`hf upload <repo> <dir> .` — uses token + API, **not git commit**, so no committer
email/identity leaks. Scrub first: local paths in `mtplx_runtime.json` (`/Users/...`), remove
`.bak` files, grep for username/email. Repo is self-contained: real weight files (no symlinks),
fixed template, model card. License: MIT (Ornith) preserved; MTP tensors are Apache-2.0 (Qwen).

## Traps hit (highest reuse value)

1. **bf16 MoE-MTP is unusably slow as a draft** — must quantize. Acceptance being high (92%)
   is a red herring; measure draft ms/token.
2. **forge skips branding on speed-loss**: if verdict is `no_mtp_depth_beat_ar`, forge writes
   `build_outcome.json` and raises instead of writing `mtplx_runtime.json` → tune.sh can't find
   the model. Fix: call forge's `_stamp_runtime_metadata` with on-disk data (don't re-forge).
   On a speed win, forge writes runtime.json itself.
3. **clonefile vs symlink**: "save disk by reusing weights" → if it lands as a *symlink* to a
   staging dir that later gets deleted, the product breaks (`Failed to open file`). Use `cp -c`
   (APFS clonefile) so the product is self-contained.
4. **forge output lands in `~/Documents/MTPLX/models/`**, not `~/.mtplx/models/`. tune.sh scans
   both; serve-webui.sh / serve-mtplx-web.sh scan only `~/.mtplx/models/` → move it there or it's
   not selectable. (serve-webui.sh now auto-detects the template system-check and picks
   local_qwen36+hybrid as a fallback, but that's local-only — published models must fix the
   template itself.)
5. **mlx_lm 0.31.3 `qwen3_5.py` sanitize lacks expert-stacking** (qwen3_moe.py has it) → forge
   convert fails "30720 parameters not in model". Port the stack logic in (backed up `.bak`).
6. **Qwen3.5 RMSNorm is delta-encoded** → MTP sidecar needs `mtplx_mtp_norm_encoding="delta"` in
   config or norms are off-by-one.
7. **tune.sh / mtplx use `.venv-mtplx`, not miniconda** — bare `python3` resolves to miniconda
   which has no mtplx module. Activate the venv first.

## Notes
- Original MTPLX has **no `/v1/responses`** → Codex CLI can't connect to upstream MTPLX (only this
  fork adds it). That's an *engine* limitation, not the model's; the model card stays
  model-only and doesn't mention Codex.
- Local memory `ornith-mtp-graft-project.md` and `~/models/MODELS.md` track the same work.
