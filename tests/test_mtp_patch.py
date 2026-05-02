from __future__ import annotations

import pytest

from mtplx.mtp_patch import MTPContract


def test_mtp_contract_reads_config_quant_defaults() -> None:
    contract = MTPContract().with_config_defaults(
        {
            "mtplx_mtp_quantization": {
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
                "prequantized": True,
            }
        }
    )

    assert contract.mtp_quant_policy == "cyankiwi"
    assert contract.mtp_quant_bits == 4
    assert contract.mtp_quant_group_size == 32
    assert contract.mtp_quant_mode == "affine"
    assert contract.mtp_prequantized is True


def test_mtp_contract_cli_bits_override_config_bits() -> None:
    contract = MTPContract(mtp_quant_bits=8).with_config_defaults(
        {
            "mtplx_mtp_quantization": {
                "policy": "cyankiwi",
                "bits": 4,
                "group_size": 32,
                "mode": "affine",
            }
        }
    )

    assert contract.mtp_quant_bits == 8
    assert contract.mtp_quant_group_size == 32
    assert contract.mtp_quant_policy == "cyankiwi"


def test_mtp_contract_rejects_unknown_quant_policy() -> None:
    with pytest.raises(ValueError, match="mtp_quant_policy"):
        MTPContract(mtp_quant_policy="mystery").validate()
