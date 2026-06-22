"""OpenAI Responses API (`POST /v1/responses`) translation layer.

MTPLX implements the Responses API the same way llama.cpp does: a Responses
request is converted into a Chat Completions request, run through the existing
chat path, and the chat result is converted back into Responses-shaped output
(both the non-streaming JSON body and the SSE event stream).

This module holds only the pure translation logic and the request model. The
routes live in `mtplx.server.openai` (`create_app`), which imports from here
and wires these helpers to the shared `chat_completions` handler. The SSE
translator takes `iter_sse_data` as a parameter so this module does not need to
import from `openai.py` (avoiding a circular import).

Mapping reference: ggml-org/llama.cpp `server_chat_convert_responses_to_chatcmpl`
(request) and `server-task.cpp` `to_json_oaicompat_resp[_stream]` (response/SSE).
"""

from __future__ import annotations

import json
import time
import uuid
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict


def _chat_models() -> tuple[Any, Any]:
    """Return (ChatMessage, ChatCompletionRequest) from the server module.

    Imported lazily to avoid a circular import: `mtplx.server.openai` imports
    this module at load time, so we cannot import its model classes at the top
    level here. By the time any request handler calls into these functions,
    `openai.py` is fully initialized.
    """

    from .openai import ChatCompletionRequest, ChatMessage

    return ChatMessage, ChatCompletionRequest


class ResponsesRequest(BaseModel):
    """OpenAI Responses API request body (`POST /v1/responses`).

    Translated into a ChatCompletionRequest and run through the shared chat
    path; see `responses_to_chat_request`. Mirrors llama.cpp's Responses
    converter: unknown fields are tolerated (extra="allow"), `input` and
    `instructions` are remapped, `previous_response_id` is rejected.
    """

    model_config = ConfigDict(extra="allow")

    model: str | None = None
    input: Any = None
    instructions: str | None = None
    max_output_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    stream: bool = False
    reasoning: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    previous_response_id: str | None = None


def responses_input_to_chat_messages(value: Any) -> list[Any]:
    """Convert a Responses `input` value into chat messages.

    Follows llama.cpp's server_chat_convert_responses_to_chatcmpl: a plain
    string becomes one user message; an array dispatches per item type
    (input messages, function_call, function_call_output, reasoning).
    """

    ChatMessage, _ = _chat_models()

    if value is None:
        raise HTTPException(status_code=400, detail="'input' is required")
    if isinstance(value, str):
        return [ChatMessage(role="user", content=value)]
    if not isinstance(value, list):
        raise HTTPException(
            status_code=400,
            detail="'input' must be a string or array of objects",
        )

    messages: list[Any] = []

    def _last_assistant() -> Any | None:
        if messages and messages[-1].role == "assistant":
            return messages[-1]
        return None

    for item in value:
        if not isinstance(item, dict):
            raise HTTPException(
                status_code=400, detail="Cannot determine type of 'item'"
            )
        item_type = item.get("type")
        role = item.get("role")
        content = item.get("content")

        # String content (with or without a role) -> input message.
        if isinstance(content, str) and item_type in (None, "message"):
            messages.append(ChatMessage(role=role or "user", content=content))
            continue

        # Input message with structured content parts.
        if role in ("user", "system", "developer") and isinstance(content, list):
            parts: list[dict[str, Any]] = []
            for part in content:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type")
                if ptype in ("input_text", "output_text", "text"):
                    parts.append({"type": "text", "text": part.get("text") or ""})
                elif ptype in ("input_image", "image_url"):
                    image_url = part.get("image_url")
                    if isinstance(image_url, dict):
                        url = image_url.get("url") or ""
                    else:
                        url = image_url or ""
                    parts.append({"type": "image_url", "image_url": {"url": url}})
                elif ptype == "input_file":
                    raise HTTPException(
                        status_code=400, detail="input_file is not supported"
                    )
            mapped_role = "system" if role == "developer" else role
            messages.append(ChatMessage(role=mapped_role, content=parts))
            continue

        # Assistant message.
        if role == "assistant" or item_type == "message":
            parts = []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type")
                    if ptype in ("output_text", "input_text", "text"):
                        parts.append({"type": "text", "text": part.get("text") or ""})
                    elif ptype == "refusal":
                        parts.append(
                            {"type": "refusal", "refusal": part.get("refusal") or ""}
                        )
            elif isinstance(content, str):
                parts.append({"type": "text", "text": content})
            prev = _last_assistant()
            if prev is not None and isinstance(prev.content, list):
                prev.content.extend(parts)
            else:
                messages.append(ChatMessage(role="assistant", content=parts))
            continue

        # Tool/function call emitted by the model.
        if item_type == "function_call":
            call = {
                "type": "function",
                "id": item.get("call_id") or item.get("id") or "",
                "function": {
                    "name": item.get("name") or "",
                    "arguments": item.get("arguments") or "",
                },
            }
            prev = _last_assistant()
            if prev is not None:
                existing = list(prev.tool_calls or [])
                existing.append(call)
                prev.tool_calls = existing
            else:
                messages.append(
                    ChatMessage(role="assistant", content="", tool_calls=[call])
                )
            continue

        # Tool result fed back in.
        if item_type == "function_call_output":
            output = item.get("output")
            if isinstance(output, list):
                text_chunks: list[str] = []
                for part in output:
                    if not isinstance(part, dict) or part.get("type") != "input_text":
                        raise HTTPException(
                            status_code=400,
                            detail="Output of tool call should be 'Input text'",
                        )
                    text_chunks.append(part.get("text") or "")
                output_text = "".join(text_chunks)
            else:
                output_text = output if isinstance(output, str) else ""
            messages.append(
                ChatMessage(
                    role="tool",
                    content=output_text,
                    tool_call_id=item.get("call_id") or "",
                )
            )
            continue

        # Reasoning trace replayed from a prior turn.
        if item_type == "reasoning":
            reasoning_text = ""
            rcontent = item.get("content")
            if isinstance(rcontent, list) and rcontent:
                first = rcontent[0]
                if isinstance(first, dict):
                    reasoning_text = first.get("text") or ""
            prev = _last_assistant()
            if prev is not None:
                prev.reasoning_content = reasoning_text  # type: ignore[attr-defined]
            else:
                msg = ChatMessage(role="assistant", content=[])
                msg.reasoning_content = reasoning_text  # type: ignore[attr-defined]
                messages.append(msg)
            continue

        raise HTTPException(status_code=400, detail="Cannot determine type of 'item'")

    return messages


def responses_tools_to_chat(tools: Any) -> list[dict[str, Any]] | None:
    """Convert Responses-style function tools into chat-completions tools."""

    if not isinstance(tools, list):
        return None
    converted: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            continue  # only function tools are supported, others are skipped
        spec = {k: v for k, v in tool.items() if k != "type"}
        spec.setdefault("strict", True)
        converted.append({"type": "function", "function": spec})
    return converted or None


def responses_to_chat_request(request: ResponsesRequest) -> Any:
    ChatMessage, ChatCompletionRequest = _chat_models()
    if request.previous_response_id:
        raise HTTPException(
            status_code=400,
            detail="MTPLX does not support 'previous_response_id'.",
        )
    messages: list[Any] = []
    if request.instructions:
        messages.append(ChatMessage(role="system", content=request.instructions))
    messages.extend(responses_input_to_chat_messages(request.input))
    reasoning_effort = None
    if isinstance(request.reasoning, dict):
        effort = request.reasoning.get("effort")
        if isinstance(effort, str):
            reasoning_effort = effort
    return ChatCompletionRequest(
        model=request.model,
        messages=messages,
        max_tokens=request.max_output_tokens,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        tools=responses_tools_to_chat(request.tools),
        tool_choice=request.tool_choice,
        reasoning_effort=reasoning_effort,
        stream=bool(request.stream),
        metadata=request.metadata,
    )


def responses_output_from_openai_message(
    message: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the Responses `output` array from an OpenAI chat message."""

    output: list[dict[str, Any]] = []
    reasoning = message.get("reasoning_content")
    if isinstance(reasoning, str) and reasoning.strip():
        output.append(
            {
                "id": "rs_" + uuid.uuid4().hex,
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": reasoning}],
                "encrypted_content": "",
                "status": "completed",
            }
        )
    text = message.get("content")
    if isinstance(text, str) and text:
        output.append(
            {
                "id": "msg_" + uuid.uuid4().hex,
                "type": "message",
                "role": message.get("role") or "assistant",
                "status": "completed",
                "content": [
                    {
                        "type": "output_text",
                        "text": text,
                        "annotations": [],
                        "logprobs": [],
                    }
                ],
            }
        )
    tool_calls = message.get("tool_calls")
    if isinstance(tool_calls, list):
        for call in tool_calls:
            if not isinstance(call, dict):
                continue
            fn = call.get("function") or {}
            output.append(
                {
                    "type": "function_call",
                    "status": "completed",
                    "arguments": fn.get("arguments") or "",
                    "call_id": "fc_" + str(call.get("id") or ""),
                    "name": fn.get("name") or "",
                }
            )
    return output


def responses_payload_from_openai(openai_payload: dict[str, Any]) -> dict[str, Any]:
    choices = openai_payload.get("choices") or []
    choice = choices[0] if choices else {}
    message = choice.get("message") or {}
    usage = openai_payload.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    now = int(time.time())
    return {
        "id": "resp_" + uuid.uuid4().hex,
        "object": "response",
        "created_at": now,
        "completed_at": now,
        "status": "completed",
        "model": openai_payload.get("model"),
        "output": responses_output_from_openai_message(message),
        "usage": {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "input_tokens_details": {"cached_tokens": 0},
        },
        "mtplx_stats": openai_payload.get("mtplx_stats"),
    }


def responses_sse(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload)}\n\n"


async def responses_stream_from_openai_sse(
    body_iterator: Any, *, model: str, iter_sse_data: Any
):
    """Translate the OpenAI chat SSE stream into Responses API events.

    `iter_sse_data` is the async SSE-frame parser from `mtplx.server.openai`,
    passed in to avoid a circular import.
    """

    response_id = "resp_" + uuid.uuid4().hex
    reasoning_id = "rs_" + uuid.uuid4().hex
    message_id = "msg_" + uuid.uuid4().hex
    reasoning_started = False
    text_started = False
    content_part_added = False
    reasoning_buffer: list[str] = []
    text_buffer: list[str] = []
    # call index -> {"call_id","name","arguments"}
    tool_calls: dict[int, dict[str, Any]] = {}
    tool_order: list[int] = []
    usage = {"input_tokens": 0, "output_tokens": 0}

    def response_envelope(status: str) -> dict[str, Any]:
        return {
            "id": response_id,
            "object": "response",
            "status": status,
            "model": model,
        }

    yield responses_sse(
        "response.created",
        {"type": "response.created", "response": response_envelope("in_progress")},
    )
    yield responses_sse(
        "response.in_progress",
        {"type": "response.in_progress", "response": response_envelope("in_progress")},
    )

    try:
        async for data in iter_sse_data(body_iterator):
            if data == "[DONE]":
                break
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                yield responses_sse(
                    "error",
                    {
                        "type": "error",
                        "message": f"failed to parse upstream SSE chunk: {exc}",
                    },
                )
                return
            if "error" in payload:
                error = payload.get("error") or {}
                yield responses_sse(
                    "error",
                    {
                        "type": "error",
                        "message": str(error.get("message") or error),
                    },
                )
                return
            if payload.get("usage"):
                upstream = payload.get("usage") or {}
                usage = {
                    "input_tokens": int(upstream.get("prompt_tokens") or 0),
                    "output_tokens": int(upstream.get("completion_tokens") or 0),
                }
            for choice in payload.get("choices") or []:
                delta = choice.get("delta") or {}

                reasoning_delta = str(delta.get("reasoning_content") or "")
                if reasoning_delta:
                    if not reasoning_started:
                        reasoning_started = True
                        yield responses_sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "item": {
                                    "id": reasoning_id,
                                    "type": "reasoning",
                                    "summary": [],
                                    "content": [],
                                    "encrypted_content": "",
                                    "status": "in_progress",
                                },
                            },
                        )
                    reasoning_buffer.append(reasoning_delta)
                    yield responses_sse(
                        "response.reasoning_text.delta",
                        {
                            "type": "response.reasoning_text.delta",
                            "item_id": reasoning_id,
                            "delta": reasoning_delta,
                        },
                    )

                text_delta = str(delta.get("content") or "")
                if text_delta:
                    if not text_started:
                        text_started = True
                        yield responses_sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "item": {
                                    "id": message_id,
                                    "type": "message",
                                    "role": "assistant",
                                    "content": [],
                                    "status": "in_progress",
                                },
                            },
                        )
                    if not content_part_added:
                        content_part_added = True
                        yield responses_sse(
                            "response.content_part.added",
                            {
                                "type": "response.content_part.added",
                                "item_id": message_id,
                                "part": {"type": "output_text", "text": ""},
                            },
                        )
                    text_buffer.append(text_delta)
                    yield responses_sse(
                        "response.output_text.delta",
                        {
                            "type": "response.output_text.delta",
                            "item_id": message_id,
                            "delta": text_delta,
                        },
                    )

                for tool_call in delta.get("tool_calls") or []:
                    if not isinstance(tool_call, dict):
                        continue
                    idx = int(tool_call.get("index") or 0)
                    fn = tool_call.get("function") or {}
                    entry = tool_calls.get(idx)
                    if entry is None:
                        entry = {
                            "call_id": "fc_" + str(tool_call.get("id") or ""),
                            "name": fn.get("name") or "",
                            "arguments": "",
                        }
                        tool_calls[idx] = entry
                        tool_order.append(idx)
                        yield responses_sse(
                            "response.output_item.added",
                            {
                                "type": "response.output_item.added",
                                "item": {
                                    "type": "function_call",
                                    "status": "in_progress",
                                    "arguments": "",
                                    "call_id": entry["call_id"],
                                    "name": entry["name"],
                                },
                            },
                        )
                    if tool_call.get("id"):
                        entry["call_id"] = "fc_" + str(tool_call.get("id"))
                    if fn.get("name"):
                        entry["name"] = fn.get("name")
                    args_delta = fn.get("arguments")
                    if args_delta:
                        entry["arguments"] += args_delta
                        yield responses_sse(
                            "response.function_call_arguments.delta",
                            {
                                "type": "response.function_call_arguments.delta",
                                "item_id": entry["call_id"],
                                "delta": args_delta,
                            },
                        )
    except Exception as exc:  # pragma: no cover - defensive stream guard
        yield responses_sse(
            "error",
            {"type": "error", "message": f"stream translation failed: {exc}"},
        )
        return

    output: list[dict[str, Any]] = []

    if reasoning_started:
        reasoning_text = "".join(reasoning_buffer)
        item = {
            "id": reasoning_id,
            "type": "reasoning",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": reasoning_text}],
            "encrypted_content": "",
        }
        output.append({**item, "status": "completed"})
        yield responses_sse(
            "response.output_item.done",
            {"type": "response.output_item.done", "item": item},
        )

    if text_started:
        full_text = "".join(text_buffer)
        yield responses_sse(
            "response.output_text.done",
            {
                "type": "response.output_text.done",
                "item_id": message_id,
                "text": full_text,
            },
        )
        content_part = {
            "type": "output_text",
            "text": full_text,
            "annotations": [],
            "logprobs": [],
        }
        yield responses_sse(
            "response.content_part.done",
            {
                "type": "response.content_part.done",
                "item_id": message_id,
                "part": content_part,
            },
        )
        message_item = {
            "type": "message",
            "status": "completed",
            "id": message_id,
            "role": "assistant",
            "content": [content_part],
        }
        output.append(message_item)
        yield responses_sse(
            "response.output_item.done",
            {"type": "response.output_item.done", "item": message_item},
        )

    for idx in tool_order:
        entry = tool_calls[idx]
        call_item = {
            "type": "function_call",
            "status": "completed",
            "arguments": entry["arguments"],
            "call_id": entry["call_id"],
            "name": entry["name"],
        }
        output.append(call_item)
        yield responses_sse(
            "response.output_item.done",
            {"type": "response.output_item.done", "item": call_item},
        )

    now = int(time.time())
    yield responses_sse(
        "response.completed",
        {
            "type": "response.completed",
            "response": {
                "id": response_id,
                "object": "response",
                "created_at": now,
                "status": "completed",
                "model": model,
                "output": output,
                "usage": {
                    "input_tokens": usage["input_tokens"],
                    "output_tokens": usage["output_tokens"],
                    "total_tokens": usage["input_tokens"] + usage["output_tokens"],
                    "input_tokens_details": {"cached_tokens": 0},
                },
            },
        },
    )
