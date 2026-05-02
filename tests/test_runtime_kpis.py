from __future__ import annotations

import json
import subprocess

from mtplx.kpi.runtime_kpis import exact_paged_attention_env, run_exactness_smoke, summarize_decode_trace


def test_summarize_decode_trace_computes_window_ratios(tmp_path):
    trace = tmp_path / "decode.jsonl"
    rows = [
        {
            "event": "decode_trace_bucket",
            "generated_tokens_delta": 80,
            "generated_tokens_total": 80,
            "elapsed_s": 1.0,
            "verify_ms_per_call_delta": 40.0,
        },
        {
            "event": "decode_trace_bucket",
            "generated_tokens_delta": 80,
            "generated_tokens_total": 160,
            "elapsed_s": 2.0,
            "verify_ms_per_call_delta": 60.0,
            "mlx_memory": {"cache_memory_bytes": 1073741824},
        },
    ]
    trace.write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    summary = summarize_decode_trace(trace)

    assert summary["available"] is True
    assert summary["first64_tok_s"] == 80.0
    assert summary["last64_tok_s"] == 40.0
    assert summary["last64_over_first64"] == 0.5
    assert summary["late_verify_ms"] == 60.0
    assert summary["cache_gib_last"] == 1.0


def test_exact_paged_attention_env_defaults_to_vector_impl():
    env = exact_paged_attention_env()

    assert env["MTPLX_VLLM_METAL_PAGED_ATTN"] == "1"
    assert env["MTPLX_VLLM_METAL_PAGED_ATTN_IMPL"] == "mlx_vector_paged"
    assert env["MTPLX_VLLM_METAL_PAGED_PARTITION_THRESHOLD"] == "2048"


def test_run_exactness_smoke_uses_vector_paged_profile(monkeypatch, tmp_path):
    seen = {}

    def fake_run(cmd, **kwargs):
        seen["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="ok")

    monkeypatch.setattr(subprocess, "run", fake_run)
    out = tmp_path / "smoke.json"

    result = run_exactness_smoke("models/example", output=out)

    assert result["passed"] is True
    assert "--attention-impl" in seen["cmd"]
    assert "mlx_vector_paged" in seen["cmd"]
    assert "--partition-threshold" in seen["cmd"]
    assert "2048" in seen["cmd"]
