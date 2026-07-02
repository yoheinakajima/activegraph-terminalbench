"""Scripted stand-in for the Anthropic client.

Satisfies loop.ModelClient without any network access, so the whole
Harbor + Docker + ActiveGraph path can be exercised end to end where no
API key is available (CI, sandboxes, unit tests).

Select it per run with:
    --ak llm_client_import_path=activegraph_harness.testing:ScriptedLLM

The script is a JSON file (path in ACTIVEGRAPH_HARNESS_SCRIPT) holding a
list of turn objects, each in the loop's response schema:
    [{"reasoning": "...", "command": "...", "done": false}, ...]
Turns are played in order; when the script runs out, the client keeps
answering done=true so the loop terminates.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from activegraph_harness.llm import LLMResult


class ScriptedLLM:
    def __init__(self, model_name: str | None = None, **_: object):
        self.model = model_name or "scripted"
        script_path = os.environ.get("ACTIVEGRAPH_HARNESS_SCRIPT")
        if not script_path:
            raise RuntimeError(
                "ScriptedLLM requires ACTIVEGRAPH_HARNESS_SCRIPT to point at a "
                "JSON file with a list of turn objects"
            )
        self._turns: list[dict] = json.loads(Path(script_path).read_text())
        self._cursor = 0

    async def complete(self, messages, *, log_event) -> LLMResult:
        if self._cursor < len(self._turns):
            turn = self._turns[self._cursor]
            self._cursor += 1
        else:
            turn = {"reasoning": "script exhausted", "command": None, "done": True}
        text = json.dumps(turn)
        return LLMResult(
            text=text,
            input_tokens=sum(len(m["content"]) // 4 for m in messages),
            output_tokens=len(text) // 4,
            latency_sec=0.0,
            model=self.model,
            retries=0,
            stop_reason="end_turn",
        )
