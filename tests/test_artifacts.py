from __future__ import annotations

import json

import numpy as np
from safetensors.numpy import save_file

from mtplx.artifacts import expected_mtp_file, inspect_model
from mtplx.constants import EXPECTED_PREQUANTIZED_MTP_KEYS


def test_expected_mtp_file_uses_extra_tensor_metadata(tmp_path):
    config = {"mlx_lm_extra_tensors": {"mtp_file": "extra-mtp.safetensors"}}
    assert expected_mtp_file(tmp_path, config) == tmp_path / "extra-mtp.safetensors"


def test_inspect_model_reports_missing_config(tmp_path):
    result = inspect_model(tmp_path)
    assert result.config_exists is False
    assert result.passes_primary_gate is False


def test_inspect_model_reads_qwen_mtp_config_without_weights(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        )
    )
    result = inspect_model(tmp_path)
    assert result.model_type == "qwen3_5"
    assert result.mtp_num_hidden_layers == 1
    assert result.mtp is not None
    assert result.mtp.exists is False


def test_qwen3_5_text_subtype_can_pass_primary_gate_when_mtp_is_valid(monkeypatch, tmp_path):
    from mtplx import artifacts
    from mtplx.artifacts import MTPInspection

    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "text_config": {
                    "model_type": "qwen3_5_text",
                    "mtp_num_hidden_layers": 1,
                    "hidden_size": 5120,
                    "num_hidden_layers": 64,
                    "vocab_size": 248320,
                },
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
            }
        )
    )

    monkeypatch.setattr(
        artifacts,
        "inspect_mtp_tensors",
        lambda *_args, **_kwargs: MTPInspection(
            mtp_file=str(tmp_path / "mtp.safetensors"),
            exists=True,
            tensor_count=15,
            missing_expected_keys=(),
        ),
    )
    assert inspect_model(tmp_path).passes_primary_gate is True


def test_prequantized_mtp_sidecar_accepts_mlx_affine_scale_bias_tensors(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "model_type": "qwen3_5",
                "mtp_num_hidden_layers": 1,
                "hidden_size": 5120,
                "num_hidden_layers": 64,
                "vocab_size": 248320,
                "mlx_lm_extra_tensors": {"mtp_file": "mtp.safetensors"},
                "mtplx_mtp_quantization": {
                    "policy": "cyankiwi",
                    "bits": 4,
                    "group_size": 32,
                    "mode": "affine",
                    "prequantized": True,
                },
            }
        )
    )
    save_file(
        {key: np.ones((1,), dtype=np.float32) for key in EXPECTED_PREQUANTIZED_MTP_KEYS},
        tmp_path / "mtp.safetensors",
    )

    result = inspect_model(tmp_path)
    assert result.mtp is not None
    assert result.mtp.sidecar_format == "prequantized-mlx-affine"
    assert result.mtp.tensor_count == 29
    assert result.mtp.missing_expected_keys == ()
    assert result.mtp.extra_keys == ()
    assert result.passes_primary_gate is True
