"""Anthropic client wrapper: retries, token accounting, timing.

One class, one method. `complete()` takes the message list produced by
build_context() (first entry may be a system message), returns the text
plus token counts and latency so the loop can log them as events.

Retries: up to MAX_RETRIES on 429, 5xx, overloaded, connection and timeout
errors, with exponential backoff. Every retry is reported through the
`log_event` callback so it lands in the trial's ActiveGraph store.

The API key comes from os.environ only. Missing key fails loudly at
construction time, before any trial container is touched.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Callable

import anthropic

DEFAULT_MODEL = "claude-sonnet-4-6"
MAX_RETRIES = 3
RETRY_BASE_DELAY_SEC = 2.0

# Signature of the event sink the loop passes in: (event_type, payload).
LogEventFn = Callable[[str, dict[str, Any]], None]


class LLMError(RuntimeError):
    """Raised when the model call fails after all retries."""


@dataclass
class LLMResult:
    text: str
    input_tokens: int
    output_tokens: int
    latency_sec: float
    model: str
    retries: int
    stop_reason: str | None


def resolve_model(model_name: str | None) -> str:
    """Model precedence: --model via Harbor, then ANTHROPIC_MODEL, then default.

    Harbor passes provider-prefixed names like 'anthropic/claude-sonnet-4-6';
    the prefix is stripped because this wrapper is Anthropic-only.
    """
    model = model_name or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
    if "/" in model:
        provider, model = model.split("/", maxsplit=1)
        if provider != "anthropic":
            raise LLMError(
                f"model {model_name!r} names provider {provider!r}, but this "
                "harness only speaks the Anthropic API"
            )
    return model


def split_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    """Separate the leading system message from the chat turns."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return "", messages


class LLMClient:
    def __init__(self, model_name: str | None = None, *, max_output_tokens: int = 4096):
        self.model = resolve_model(model_name)
        self.max_output_tokens = max_output_tokens
        # Claude Code on the web reserves the ANTHROPIC_API_KEY name in its
        # environment settings (provider auth is host-managed there), so
        # accept ANTHROP_API_KEY as a fallback for sandbox runs.
        api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get(
            "ANTHROP_API_KEY"
        )
        if not api_key:
            raise LLMError(
                "ANTHROPIC_API_KEY is not set (nor the ANTHROP_API_KEY "
                "fallback). Export it before running: "
                "export ANTHROPIC_API_KEY=sk-ant-..."
            )
        # The SDK's built-in retries are disabled so every retry goes through
        # this wrapper and gets logged as an llm_retry event.
        self._client = anthropic.AsyncAnthropic(api_key=api_key, max_retries=0)

    async def complete(
        self, messages: list[dict[str, str]], *, log_event: LogEventFn
    ) -> LLMResult:
        system, turns = split_system(messages)
        last_error: Exception | None = None
        for attempt in range(MAX_RETRIES + 1):
            started = time.monotonic()
            try:
                response = await self._client.messages.create(
                    model=self.model,
                    max_tokens=self.max_output_tokens,
                    system=system,
                    messages=turns,
                )
            except (anthropic.APIStatusError, anthropic.APIConnectionError) as exc:
                # Retry 429, all 5xx (including 529 overloaded), and
                # connection/timeout errors. Anything else (401, 400, ...) is
                # a caller problem: fail immediately and loudly.
                status = getattr(exc, "status_code", None)
                retryable = status is None or status == 429 or status >= 500
                if not retryable:
                    raise
                last_error = exc
                delay = RETRY_BASE_DELAY_SEC * (2**attempt)
                log_event(
                    "llm_retry",
                    {
                        "attempt": attempt + 1,
                        "max_retries": MAX_RETRIES,
                        "error": f"{type(exc).__name__}: {exc}",
                        "delay_sec": delay,
                        "latency_sec": round(time.monotonic() - started, 3),
                    },
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(delay)
                continue

            latency = time.monotonic() - started
            text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return LLMResult(
                text=text,
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                latency_sec=round(latency, 3),
                model=response.model,
                retries=attempt,
                stop_reason=response.stop_reason,
            )

        raise LLMError(
            f"model call failed after {MAX_RETRIES + 1} attempts: {last_error}"
        ) from last_error
