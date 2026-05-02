# Development

```bash
python -m pip install -e ".[dev,server]"
python -m pytest tests/test_no_mlx_imports.py tests/test_public_cli.py tests/test_runtime_kpis.py
python -m build
scripts/fresh_venv_smoke.sh
```

Keep generated artifacts, model weights, and local credentials out of Git. The release repository is a product export, not a research workspace dump.
