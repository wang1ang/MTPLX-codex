#!/usr/bin/env python3
"""Extract the mtp.* tensors verbatim from a sharded Qwen3.5 MoE safetensors
repo (the official MTP head that already uses MTPLX-style mtp.* key names) and
write an MLX-format mtp.safetensors sidecar.

Unlike the GGUF path (extract_mtp.py), the source here is ALREADY in HF/MLX key
convention with NUMBERED experts (mtp.layers.0.mlp.experts.<E>.{gate,up,down}_proj.weight).
MTPLX's runtime (mtplx.expert_layout.stack_numbered_experts, called from
mtp_patch._stack_mtp_moe_experts) stacks these numbered experts into the
switch_mlp layout at load time, so we keep them verbatim -- no fuse, no
transpose, no rename.

Norms: kept as float32; linear/expert weights cast to bfloat16. We print norm
means so the operator can decide on mtplx_mtp_norm_encoding (delta vs real).
"""
import json
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
from safetensors import safe_open


def main(src_dir, out_path):
    src = Path(src_dir)
    idx = json.loads((src / "model.safetensors.index.json").read_text())
    wm = idx["weight_map"]
    mtp_keys = sorted(k for k in wm if k.startswith("mtp."))
    if not mtp_keys:
        raise SystemExit("no mtp.* keys in index")
    print(f"# {len(mtp_keys)} mtp.* tensors across shards")

    # which shards hold mtp keys
    shards = sorted({wm[k] for k in mtp_keys})
    out = {}
    norm_report = []
    for shard in shards:
        loaded = mx.load(str(src / shard))  # full shard, mlx native bf16
        for k in mtp_keys:
            if wm[k] != shard:
                continue
            arr = loaded[k]
            if arr.ndim == 1:
                f32 = arr.astype(mx.float32)
                out[k] = f32  # norms stay f32
                norm_report.append((k, float(f32.mean()), float(f32.min()), float(f32.max())))
            else:
                out[k] = arr.astype(mx.bfloat16)
        del loaded
        mx.clear_cache()

    mx.eval(list(out.values()))
    mx.save_safetensors(str(out_path), out, metadata={"format": "mlx"})
    print(f"wrote {out_path} with {len(out)} tensors")

    print("\n# norm (1-D) tensors mean/min/max:")
    for name, mean, lo, hi in norm_report:
        print(f"  {name}: mean={mean:.4f} min={lo:.4f} max={hi:.4f}")
    means = [m for _, m, _, _ in norm_report]
    if means and max(means) < 0.3:
        print("\n!! WARNING: norm means near 0 -> looks DELTA-encoded. "
              "Set mtplx_mtp_norm_encoding='delta' in assembled config.")
    else:
        print("\n# norm means look like REAL values (~1). Do NOT set mtplx_mtp_norm_encoding.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1], sys.argv[2])
