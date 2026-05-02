"""MTPLX: native Qwen3.6 MTP experiments on MLX."""

from __future__ import annotations

from typing import Any

__all__ = ["MTPLXRuntime", "load"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from .runtime import MTPLXRuntime, load

        exports = {"MTPLXRuntime": MTPLXRuntime, "load": load}
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
