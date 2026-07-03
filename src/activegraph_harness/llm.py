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
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    latency_sec: float
    model: str
    retries: int
    stop_reason: str | None

    @property
    def total_input_tokens(self) -> int:
        """All prompt tokens the model actually saw. The API reports
        input_tokens as only the tokens after the last cache breakpoint."""
        return (
            self.input_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )


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


def split_system(messages: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    """Separate the leading system message from the chat turns."""
    if messages and messages[0]["role"] == "system":
        return messages[0]["content"], messages[1:]
    return "", messages


def apply_cache_control(system: Any, turns: list[dict[str, Any]]) -> tuple[Any, list[dict[str, Any]]]:
    """Place prompt-cache breakpoints on the request.

    Breakpoint budget (API maximum is 4 per request; this uses 2):
    1. The system prompt, which is identical every turn of a trial.
    2. The last STABLE text block of the final message, so each turn reads
       the whole prior transcript from cache and writes only the new tail.

    build_context() marks per-turn volatile text (the budget countdown, the
    v2 retrieved slice) by putting it in blocks AFTER the last stable block
    of the final message; those blocks stay behind the breakpoint and are
    never cached. Messages whose content is a plain string are treated as
    one stable block. Earlier messages are never marked: the server matches
    the request prefix against the cache entry written at the previous
    turn's breakpoint, so one moving breakpoint is enough.

    Content blocks below the model's minimum cacheable prefix (1024 tokens
    on Sonnet, 4096 on Haiku 4.5) are silently not cached by the API; both
    usage counters just read 0. That is expected on small early turns.
    """
    system_blocks = [
        {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
    ]
    cached_turns = []
    for turn in turns:
        turn = dict(turn)
        if not isinstance(turn["content"], str):
            # Strip the harness-internal volatile flag; the API rejects
            # unknown block fields.
            turn["content"] = [
                {k: v for k, v in block.items() if k != "volatile"}
                for block in turn["content"]
            ]
        cached_turns.append(turn)
    if not cached_turns:
        return system_blocks, cached_turns

    final = cached_turns[-1]
    original_content = turns[-1]["content"]
    if isinstance(original_content, str):
        blocks = [{"type": "text", "text": original_content}]
        stable_indices = [0]
    else:
        blocks = final["content"]
        stable_indices = [
            i
            for i, block in enumerate(original_content)
            if not block.get("volatile", False)
        ]
    if stable_indices:
        blocks[stable_indices[-1]] = {
            **blocks[stable_indices[-1]],
            "cache_control": {"type": "ephemeral"},
        }
    final["content"] = blocks
    return system_blocks, cached_turns


class LLMClient:
    def __init__(
        self,
        model_name: str | None = None,
        *,
        max_output_tokens: int = 4096,
        enable_prompt_caching: bool = True,
    ):
        self.model = resolve_model(model_name)
        self.max_output_tokens = max_output_tokens
        self.enable_prompt_caching = enable_prompt_caching
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
        if self.enable_prompt_caching:
            system, turns = apply_cache_control(system, turns)
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
                # None when the request had no cache_control breakpoints.
                cache_creation_input_tokens=response.usage.cache_creation_input_tokens
                or 0,
                cache_read_input_tokens=response.usage.cache_read_input_tokens or 0,
                latency_sec=round(latency, 3),
                model=response.model,
                retries=attempt,
                stop_reason=response.stop_reason,
            )

        raise LLMError(
            f"model call failed after {MAX_RETRIES + 1} attempts: {last_error}"
        ) from last_error
