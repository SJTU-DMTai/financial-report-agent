# -*- coding: utf-8 -*-
from __future__ import annotations

import contextvars
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, AsyncGenerator

try:
    import tiktoken
except Exception:  # pragma: no cover - optional dependency fallback
    tiktoken = None

from agentscope.model import ChatModelBase, ChatResponse


_COMPONENT = contextvars.ContextVar("fra_token_component", default="")
_CALL_KIND = contextvars.ContextVar("fra_token_call_kind", default="")
_WRITE_LOCK = threading.Lock()
_ENCODING = None


def set_token_context(component: str, call_kind: str = "") -> tuple[contextvars.Token, contextvars.Token]:
    component_token = _COMPONENT.set(component or "")
    call_kind_token = _CALL_KIND.set(call_kind or "")
    return component_token, call_kind_token


def reset_token_context(tokens: tuple[contextvars.Token, contextvars.Token]) -> None:
    component_token, call_kind_token = tokens
    _COMPONENT.reset(component_token)
    _CALL_KIND.reset(call_kind_token)


def token_tracking_enabled() -> bool:
    if os.getenv("FRA_TOKEN_TRACKING", "").strip() == "0":
        return False
    return bool(os.getenv("FRA_TOKEN_TRACKING_FILE", "").strip())


def token_tracking_path() -> Path | None:
    raw = os.getenv("FRA_TOKEN_TRACKING_FILE", "").strip()
    if not raw:
        return None
    return Path(raw)


def estimate_tokens(value: Any) -> int:
    text = _stable_text(value)
    if not text:
        return 0
    encoding = _token_encoding()
    if encoding is None:
        return max(1, len(text) // 4)
    return len(encoding.encode(text))


def track_tool_result(
    tool_name: str,
    text: Any,
    metadata: dict[str, Any] | None = None,
) -> None:
    if not token_tracking_enabled():
        return
    text_value = _stable_text(text)
    event = {
        "event": "tool_result",
        "component": _COMPONENT.get(),
        "call_kind": _CALL_KIND.get(),
        "tool_name": tool_name,
        "text_chars": len(text_value),
        "text_tokens_est": estimate_tokens(text_value),
        "metadata": metadata or {},
    }
    append_token_event(event)


def append_token_event(event: dict[str, Any]) -> None:
    path = token_tracking_path()
    if path is None:
        return
    payload = {
        "ts": time.time(),
        "pid": os.getpid(),
        **event,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, default=str)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as file:
            file.write(line + "\n")


class MeteredChatModel(ChatModelBase):
    def __init__(self, inner: ChatModelBase) -> None:
        super().__init__(
            model_name=str(getattr(inner, "model_name", inner.__class__.__name__)),
            stream=bool(getattr(inner, "stream", False)),
        )
        self.inner = inner

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __setattr__(self, name: str, value: Any) -> None:
        if name in {"inner", "model_name", "stream"}:
            object.__setattr__(self, name, value)
            if name == "model_name" and "inner" in self.__dict__:
                setattr(self.inner, name, value)
            return
        if "inner" in self.__dict__ and hasattr(self.inner, name):
            setattr(self.inner, name, value)
            return
        object.__setattr__(self, name, value)

    async def __call__(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        if "inner" in self.__dict__:
            self.model_name = str(getattr(self.inner, "model_name", self.model_name))
            self.stream = bool(getattr(self.inner, "stream", self.stream))
        start = time.perf_counter()
        messages = _extract_messages(args, kwargs)
        input_tokens_est = estimate_tokens(messages)
        input_chars = len(_stable_text(messages))
        message_count = len(messages) if isinstance(messages, list) else 0
        response = await self.inner(*args, **kwargs)
        if _is_async_iterable(response):
            return _track_stream_response(self, response, start, input_tokens_est, input_chars, message_count)
        _track_chat_response(self, response, start, input_tokens_est, input_chars, message_count)
        return response


async def _track_stream_response(
    model: MeteredChatModel,
    response: AsyncGenerator[ChatResponse, None],
    start: float,
    input_tokens_est: int,
    input_chars: int,
    message_count: int,
) -> AsyncGenerator[ChatResponse, None]:
    last_response = None
    async for chunk in response:
        last_response = chunk
        yield chunk
    if last_response is not None:
        _track_chat_response(model, last_response, start, input_tokens_est, input_chars, message_count)


def _track_chat_response(
    model: MeteredChatModel,
    response: Any,
    start: float,
    input_tokens_est: int,
    input_chars: int,
    message_count: int,
) -> None:
    if not token_tracking_enabled():
        return
    usage = getattr(response, "usage", None)
    output_text = _response_text(response)
    output_tokens_est = estimate_tokens(output_text)
    event = {
        "event": "llm_call",
        "component": _COMPONENT.get() or "unscoped",
        "call_kind": _CALL_KIND.get(),
        "model": str(getattr(model, "model_name", "")),
        "input_tokens_est": input_tokens_est,
        "output_tokens_est": output_tokens_est,
        "input_tokens_api": _usage_value(usage, "input_tokens"),
        "output_tokens_api": _usage_value(usage, "output_tokens"),
        "input_chars": input_chars,
        "output_chars": len(output_text),
        "message_count": message_count,
        "elapsed_ms": int((time.perf_counter() - start) * 1000),
    }
    append_token_event(event)


def _extract_messages(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    if args:
        return args[0]
    return kwargs.get("messages", [])


def _is_async_iterable(value: Any) -> bool:
    try:
        getattr(value, "__aiter__")
    except (AttributeError, KeyError):
        return False
    return True


def _response_text(response: Any) -> str:
    if response is None:
        return ""
    content = getattr(response, "content", None)
    if content is None:
        return _stable_text(response)
    return _stable_text(content)


def _usage_value(usage: Any, name: str) -> int | None:
    if usage is None:
        return None
    names = [name]
    if name == "input_tokens":
        names.extend(["prompt_tokens", "input_token_count"])
    elif name == "output_tokens":
        names.extend(["completion_tokens", "output_token_count"])
    value = None
    for item in names:
        value = getattr(usage, item, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(item)
        if value is not None:
            break
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _stable_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)
    except Exception:
        return str(value)


def _token_encoding():
    global _ENCODING
    if _ENCODING is not None:
        return _ENCODING
    if tiktoken is None:
        return None
    _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING
