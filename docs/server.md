# Server

The v0.1 server target is OpenAI-compatible local serving.

```bash
mtplx serve --host 127.0.0.1 --port 8000
```

Planned endpoints:

- `GET /health`
- `GET /metrics`
- `GET /v1/models`
- `POST /v1/chat/completions`
- `POST /v1/completions`

Binding to a non-localhost host should require an API key.
