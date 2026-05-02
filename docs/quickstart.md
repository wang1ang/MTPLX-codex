# Quickstart

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
python -m pip install mtplx

mtplx doctor --json
mtplx init
mtplx inspect model --model /path/to/model --json
```

The commands above are no-MLX-safe except generation and serving. A missing MLX runtime should appear in `doctor` as an actionable dependency issue, not a traceback.

After a verified model is available:

```bash
mtplx run "hello"
mtplx chat
mtplx serve --port 8000
```
