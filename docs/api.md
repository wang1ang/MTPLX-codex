# API

MTPLX v0.1 targets OpenAI-compatible local serving first.

## `GET /health`

Reports model load state, profile, exactness baseline, MLX/runtime information, fan mode, and warmup status.

## `GET /metrics`

Reports runtime KPIs as JSON or Prometheus-style text, depending on server configuration.

## `GET /v1/models`

Lists cached and active models.

## `POST /v1/chat/completions`

OpenAI-compatible chat completions. Streaming uses server-sent events.

## `POST /v1/completions`

Legacy OpenAI completions.

Anthropic `/v1/messages` compatibility is deferred until the OpenAI baseline is stable.
