# Install MTPLX

MTPLX is preview software for Apple Silicon Macs.

## Requirements

- Apple Silicon Mac
- Python 3.11, 3.12, or 3.13
- macOS with MLX support
- Enough disk for the selected model

## Install

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install mtplx
```

For local development:

```bash
python -m pip install -e ".[dev,server]"
```

## Runtime Dependencies

`mtplx --help`, `mtplx doctor`, `mtplx inspect`, and `mtplx init` are designed to work even before MLX is installed. Generation and serving require MLX and a verified model.

The v0.1 default dependency path uses vanilla `mlx`. The opt-in `performance-cold` profile may require the MTPLX MLX fork until the custom-kernel work is upstreamed or extracted.

## Optional Thermal Tools

`--max` is opt-in. If ThermalForge or another supported fan-control CLI is not present, MTPLX should print instructions and continue without fan control. It must not silently enable spin-loop or clock-anchor modes.
