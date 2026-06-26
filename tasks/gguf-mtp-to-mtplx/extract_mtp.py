#!/usr/bin/env python3
"""Extract the nextn/MTP head tensors from a (partial) Qwen3.5-family GGUF and
write an MTPLX-compatible mtp.safetensors (MLX format).

Generic over the model: the MTP block index is read from the plan's metadata
(block_count - 1) rather than hard-coded, and the GGUF->MTPLX key mapping is
applied per-tensor by suffix.

Usage:
  python3 extract_mtp.py <partial.gguf> <plan.json> <out/mtp.safetensors>

Assumptions (verified for empero-ai/Qwythos-9B Qwen3.5 family):
  * Source is an *-MTP-BF16.gguf (tensors are BF16/F32, no dequant needed).
  * GGUF dims are reversed vs row-major; after reversing, 2-D linear weights
    are already [out, in] (HF/MTPLX convention) -> no transpose.
  * RMSNorm weights in this GGUF are *real values* (~1), NOT delta-encoded, so
    we keep them as-is and the assembled config must NOT set
    mtplx_mtp_norm_encoding. (If a future GGUF stores norms as weight-1, set
    that flag in config instead of changing this script.)
"""
import json
import sys

import numpy as np
import mlx.core as mx

# Map a GGUF nextn/attn tensor (suffix after "blk.<MTP>.") to the MTPLX key.
# Keys here are the part of the GGUF name after the block prefix.
SUFFIX_MAP = {
    "nextn.eh_proj.weight":          "mtp.fc.weight",
    "nextn.enorm.weight":            "mtp.pre_fc_norm_embedding.weight",
    "nextn.hnorm.weight":            "mtp.pre_fc_norm_hidden.weight",
    "nextn.shared_head_norm.weight": "mtp.norm.weight",
    "attn_norm.weight":              "mtp.layers.0.input_layernorm.weight",
    "post_attention_norm.weight":    "mtp.layers.0.post_attention_layernorm.weight",
    "attn_q.weight":                 "mtp.layers.0.self_attn.q_proj.weight",
    "attn_k.weight":                 "mtp.layers.0.self_attn.k_proj.weight",
    "attn_v.weight":                 "mtp.layers.0.self_attn.v_proj.weight",
    "attn_output.weight":            "mtp.layers.0.self_attn.o_proj.weight",
    "attn_q_norm.weight":            "mtp.layers.0.self_attn.q_norm.weight",
    "attn_k_norm.weight":            "mtp.layers.0.self_attn.k_norm.weight",
    "ffn_gate.weight":               "mtp.layers.0.mlp.gate_proj.weight",
    "ffn_up.weight":                 "mtp.layers.0.mlp.up_proj.weight",
    "ffn_down.weight":               "mtp.layers.0.mlp.down_proj.weight",
}


def read_bf16(buf, off, nelem):
    raw = np.frombuffer(buf, dtype=np.uint16, count=nelem, offset=off)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def read_f32(buf, off, nelem):
    return np.frombuffer(buf, dtype="<f4", count=nelem, offset=off).copy()


def main(partial_path, plan_path, out_path):
    plan = json.load(open(plan_path))
    with open(partial_path, "rb") as f:
        buf = f.read()

    # plan["tensors"] only contains the MTP block (blk.<N>.*). Derive suffix.
    block_prefix = None
    for t in plan["tensors"]:
        parts = t["name"].split(".", 2)  # blk.<N>.<rest>
        if len(parts) == 3 and parts[0] == "blk":
            block_prefix = f"{parts[0]}.{parts[1]}."
            break
    if block_prefix is None:
        raise SystemExit("could not determine MTP block prefix from plan")
    print(f"# MTP block prefix: {block_prefix}")

    by_suffix = {}
    for t in plan["tensors"]:
        suffix = t["name"][len(block_prefix):]
        by_suffix[suffix] = t

    missing = [s for s in SUFFIX_MAP if s not in by_suffix]
    if missing:
        raise SystemExit(f"plan missing tensors (suffix): {missing}")

    out = {}
    for suffix, mkey in SUFFIX_MAP.items():
        t = by_suffix[suffix]
        dt, dims, off, nbytes = t["dtype"], t["dims"], t["abs_offset"], t["nbytes"]
        if dt not in ("BF16", "F32"):
            raise SystemExit(f"{t['name']} is {dt}; use the *-MTP-BF16.gguf (no dequant path here)")
        if off + nbytes > len(buf):
            raise SystemExit(f"{t['name']}: need byte {off+nbytes:,}, file only {len(buf):,} -- download more")
        nelem = 1
        for d in dims:
            nelem *= d
        flat = read_bf16(buf, off, nelem) if dt == "BF16" else read_f32(buf, off, nelem)
        arr = flat.reshape(list(reversed(dims)))
        if arr.ndim == 1:
            out[mkey] = mx.array(arr.astype(np.float32))  # norms stay F32
        else:
            out[mkey] = mx.array(arr.astype(np.float32)).astype(mx.bfloat16)

    mx.eval(list(out.values()))
    out_path = str(out_path)
    mx.save_safetensors(out_path, out, metadata={"format": "mlx"})
    print(f"wrote {out_path} with {len(out)} tensors")
    for k in sorted(out):
        print(f"  {k}\t{tuple(out[k].shape)}\t{out[k].dtype}")

    # sanity: norm distribution (should be ~1, i.e. NOT delta-encoded)
    norm_means = [float(mx.mean(out[k].astype(mx.float32)).item())
                  for k in out if out[k].ndim == 1]
    if norm_means and max(norm_means) < 0.3:
        print("\n!! WARNING: norm means near 0 -> looks delta-encoded. "
              "Set mtplx_mtp_norm_encoding='delta' in the assembled config.")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
