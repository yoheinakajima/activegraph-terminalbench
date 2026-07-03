"""The ReAct loop. Model-agnostic: one turn = one model call = at most one
shell command, until the model says done or a budget runs out.

Everything that happens is appended to the trial's ActiveGraph store as it
happens, and mirrored into the typed graph (Step, Command, Observation,
Error objects with edges). The loop reports each finished turn through the
`record_turn` callback so the agent can populate Harbor's context
incrementally; a timeout mid-run still leaves a usable partial trajectory.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from harbor.environments.base import BaseEnvironment

from activegraph_harness import events
from activegraph_harness.context import build_context
from activegraph_harness.events import TrialLog
from activegraph_harness.llm import LLMResult

DEFAULT_MAX_STEPS = 60
DEFAULT_COMMAND_TIMEOUT_SEC = 180
# Harbor does not tell the agent the task's timeout, so the soft budget
# defaults to the modal TB2 agent timeout (900s) minus 60s of headroom for
# export and shutdown. Raise it per run with --ak wall_clock_budget_sec=...
# for the long tasks (TB2 has tasks up to 12000s).
DEFAULT_WALL_CLOCK_BUDGET_SEC = 840.0
DEFAULT_MAX_FEEDBACK_CHARS = 4000

TRUNCATION_MARKER = "\n[... output truncated: {omitted} of {total} chars omitted ...]\n"


class ParseError(ValueError):
    """The model response was not the required JSON object."""


class ModelClient(Protocol):
    """What the loop needs from an LLM wrapper. llm.LLMClient satisfies it;
    tests satisfy it with a scripted stand-in."""

    model: str

    async def complete(
        self, messages: list[dict[str, str]], *, log_event
    ) -> LLMResult: ...


@dataclass
class Budgets:
    max_steps: int = DEFAULT_MAX_STEPS
    wall_clock_sec: float = DEFAULT_WALL_CLOCK_BUDGET_SEC
    command_timeout_sec: int = DEFAULT_COMMAND_TIMEOUT_SEC
    max_feedback_chars: int = DEFAULT_MAX_FEEDBACK_CHARS
    started_at: float = field(default_factory=time.monotonic)

    def elapsed_sec(self) -> float:
        return time.monotonic() - self.started_at

    def remaining_sec(self) -> float:
        return self.wall_clock_sec - self.elapsed_sec()

    def exhausted_reason(self, step: int) -> str | None:
        if step >= self.max_steps:
            return f"max steps reached ({self.max_steps})"
        if self.remaining_sec() <= 0:
            return f"wall clock budget exhausted ({int(self.wall_clock_sec)}s)"
        return None

    def status_line(self, step: int) -> str:
        return (
            f"step {step + 1} of {self.max_steps}, "
            f"about {max(0, int(self.remaining_sec()))}s of "
            f"{int(self.wall_clock_sec)}s wall clock remaining"
        )


@dataclass
class TurnRecord:
    """Everything the agent needs to mirror one turn into Harbor's context."""

    step: int
    reasoning: str
    raw_response: str
    command: str | None
    done: bool
    parse_failed: bool
    exit_code: int | None
    feedback_text: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    latency_sec: float
    command_wall_time_sec: float | None


@dataclass
class LoopResult:
    finish_reason: str  # task_finished | budget_exhausted
    n_steps: int
    n_commands: int
    n_errors: int
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


def parse_response(text: str) -> dict[str, Any]:
    """Parse the model's JSON turn. Strict on shape, tolerant on wrapping:
    accepts code fences or prose around the object by slicing from the
    first '{' to the last '}'."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end <= start:
        raise ParseError("no JSON object found in response")
    try:
        parsed = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ParseError(f"invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ParseError("response JSON is not an object")
    missing = {"reasoning", "command", "done"} - parsed.keys()
    if missing:
        raise ParseError(f"response JSON missing keys: {sorted(missing)}")
    if not isinstance(parsed["reasoning"], str):
        raise ParseError("'reasoning' must be a string")
    if parsed["command"] is not None and not isinstance(parsed["command"], str):
        raise ParseError("'command' must be a string or null")
    if not isinstance(parsed["done"], bool):
        raise ParseError("'done' must be a boolean")
    return parsed


def truncate_middle(text: str, max_chars: int) -> tuple[str, bool]:
    """Head+tail truncation with an explicit marker. Returns (text, truncated)."""
    if len(text) <= max_chars:
        return text, False
    half = max_chars // 2
    omitted = len(text) - 2 * half
    marker = TRUNCATION_MARKER.format(omitted=omitted, total=len(text))
    return text[:half] + marker + text[-half:], True


async def run_loop(
    environment: BaseEnvironment,
    instruction: str,
    log: TrialLog,
    llm: ModelClient,
    budgets: Budgets,
    record_turn: Callable[[TurnRecord], None],
) -> LoopResult:
    """Run the ReAct loop until done or budget exhaustion.

    Raises on unrecoverable errors (LLM failure after retries, store
    failure); the caller logs and exports in its finally block. Command
    failures and parse failures are not unrecoverable: they become events
    and feedback for the next turn.
    """
    task_object_id = events.add_task_object(log, instruction)
    previous_step_id: str | None = None
    n_commands = 0
    n_errors = 0
    total_input_tokens = 0
    total_output_tokens = 0
    total_cache_creation_tokens = 0
    total_cache_read_tokens = 0
    consecutive_parse_failures = 0
    step = 0

    while True:
        reason = budgets.exhausted_reason(step)
        if reason is not None:
            events.append_event(
                log,
                "budget_exhausted",
                step=step,
                payload={"reason": reason, "elapsed_sec": round(budgets.elapsed_sec(), 3)},
            )
            return LoopResult(
                finish_reason="budget_exhausted",
                n_steps=step,
                n_commands=n_commands,
                n_errors=n_errors,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            )

        # 1. Assemble context (the one retrieval seam).
        messages = build_context(log, instruction, step, budgets)

        # 2. Call the model. LLM retries are logged from inside the client.
        def log_llm_event(event_type: str, payload: dict[str, Any], _step=step) -> None:
            events.append_event(log, event_type, step=_step, payload=payload)

        try:
            result = await llm.complete(messages, log_event=log_llm_event)
        except Exception as exc:
            n_errors += 1
            events.append_event(
                log,
                "error_raised",
                step=step,
                payload={"kind": type(exc).__name__, "message": str(exc), "fatal": True},
            )
            events.add_error_object(log, None, kind=type(exc).__name__, message=str(exc))
            raise

        total_input_tokens += result.input_tokens
        total_output_tokens += result.output_tokens
        total_cache_creation_tokens += result.cache_creation_input_tokens
        total_cache_read_tokens += result.cache_read_input_tokens

        # 3. Parse defensively.
        parsed: dict[str, Any] | None = None
        parse_message = ""
        try:
            parsed = parse_response(result.text)
            consecutive_parse_failures = 0
        except ParseError as exc:
            consecutive_parse_failures += 1
            parse_message = str(exc)

        reasoning = parsed["reasoning"] if parsed else ""
        command = parsed["command"] if parsed else None
        done = parsed["done"] if parsed else False

        events.append_event(
            log,
            "model_turn",
            step=step,
            payload={
                "reasoning": reasoning,
                "raw_response": result.text,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "cache_creation_input_tokens": result.cache_creation_input_tokens,
                "cache_read_input_tokens": result.cache_read_input_tokens,
                "latency_sec": result.latency_sec,
                "retries": result.retries,
                "stop_reason": result.stop_reason,
                "response_model": result.model,
                "parse_failed": parsed is None,
            },
        )
        step_object_id = events.add_step_object(log, step, reasoning, previous_step_id)
        previous_step_id = step_object_id

        if parsed is None:
            n_errors += 1
            if consecutive_parse_failures >= 2:
                # Second failure in a row: treat as a no-op turn.
                feedback = (
                    "Your previous two responses could not be parsed. This turn "
                    "was treated as a no-op. Respond with exactly one JSON object: "
                    '{"reasoning": str, "command": str or null, "done": bool}'
                )
                event_type = "noop_turn"
                consecutive_parse_failures = 0
            else:
                feedback = (
                    f"Your response could not be parsed: {parse_message}. "
                    "Respond again with exactly one JSON object: "
                    '{"reasoning": str, "command": str or null, "done": bool}'
                )
                event_type = "parse_error"
            events.append_event(
                log,
                event_type,
                step=step,
                payload={"error": parse_message, "feedback_text": feedback},
            )
            events.add_error_object(
                log, None, kind="ParseError", message=parse_message
            )
            record_turn(
                TurnRecord(
                    step=step,
                    reasoning="",
                    raw_response=result.text,
                    command=None,
                    done=False,
                    parse_failed=True,
                    exit_code=None,
                    feedback_text=feedback,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cache_creation_input_tokens=result.cache_creation_input_tokens,
                    cache_read_input_tokens=result.cache_read_input_tokens,
                    latency_sec=result.latency_sec,
                    command_wall_time_sec=None,
                )
            )
            step += 1
            continue

        # 4. Done with no command: finish.
        if done and command is None:
            events.append_event(
                log,
                "task_finished",
                step=step,
                payload={
                    "reasoning": reasoning,
                    "elapsed_sec": round(budgets.elapsed_sec(), 3),
                },
            )
            record_turn(
                TurnRecord(
                    step=step,
                    reasoning=reasoning,
                    raw_response=result.text,
                    command=None,
                    done=True,
                    parse_failed=False,
                    exit_code=None,
                    feedback_text="",
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cache_creation_input_tokens=result.cache_creation_input_tokens,
                    cache_read_input_tokens=result.cache_read_input_tokens,
                    latency_sec=result.latency_sec,
                    command_wall_time_sec=None,
                )
            )
            return LoopResult(
                finish_reason="task_finished",
                n_steps=step + 1,
                n_commands=n_commands,
                n_errors=n_errors,
                input_tokens=total_input_tokens,
                output_tokens=total_output_tokens,
                cache_creation_input_tokens=total_cache_creation_tokens,
                cache_read_input_tokens=total_cache_read_tokens,
            )

        # 5. No command and not done: no-op turn, tell the model.
        if command is None:
            feedback = (
                "You returned no command and done was false, so nothing was "
                "executed. Issue a command, or set done to true to finish."
            )
            events.append_event(
                log,
                "noop_turn",
                step=step,
                payload={"feedback_text": feedback},
            )
            record_turn(
                TurnRecord(
                    step=step,
                    reasoning=reasoning,
                    raw_response=result.text,
                    command=None,
                    done=False,
                    parse_failed=False,
                    exit_code=None,
                    feedback_text=feedback,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    cache_creation_input_tokens=result.cache_creation_input_tokens,
                    cache_read_input_tokens=result.cache_read_input_tokens,
                    latency_sec=result.latency_sec,
                    command_wall_time_sec=None,
                )
            )
            step += 1
            continue

        # 6. Execute the command in the task container.
        command_object_id = events.add_command_object(
            log, step_object_id, command, budgets.command_timeout_sec
        )
        command_started = time.monotonic()
        exec_error: str | None = None
        try:
            exec_result = await environment.exec(
                command, timeout_sec=budgets.command_timeout_sec
            )
            exit_code = exec_result.return_code
            stdout = exec_result.stdout or ""
            stderr = exec_result.stderr or ""
        except Exception as exc:
            # Timeouts and docker failures surface here (RuntimeError from
            # the environment). Not fatal: log it and feed it back.
            exec_error = f"{type(exc).__name__}: {exc}"
            exit_code = -1
            stdout = ""
            stderr = exec_error
        command_wall_time = round(time.monotonic() - command_started, 3)
        n_commands += 1

        events.append_event(
            log,
            "command_executed",
            step=step,
            payload={
                "command": command,
                "exit_code": exit_code,
                "wall_time_sec": command_wall_time,
                "timeout_sec": budgets.command_timeout_sec,
                "exec_error": exec_error,
            },
        )

        # 7. Observe. Full output goes into the event; the model gets a
        # head+tail truncated version.
        combined = stdout
        if stderr:
            combined = f"{stdout}\n[stderr]\n{stderr}" if stdout else f"[stderr]\n{stderr}"
        body, truncated = truncate_middle(combined, budgets.max_feedback_chars)
        feedback = f"exit code: {exit_code}\n{body}" if body else f"exit code: {exit_code}\n(no output)"

        events.append_event(
            log,
            "output_observed",
            step=step,
            payload={
                "exit_code": exit_code,
                "stdout": stdout,
                "stderr": stderr,
                "truncated": truncated,
                "feedback_text": feedback,
            },
        )
        observation_object_id = events.add_observation_object(
            log,
            command_object_id,
            exit_code=exit_code,
            stdout=stdout,
            stderr=stderr,
            wall_time_sec=command_wall_time,
            truncated=truncated,
            preview=body[:500],
        )

        if exit_code != 0:
            n_errors += 1
            message = exec_error or (stderr[:2000] or f"command exited {exit_code}")
            events.append_event(
                log,
                "error_raised",
                step=step,
                payload={
                    "kind": "ExecError" if exec_error else "NonZeroExit",
                    "message": message,
                    "fatal": False,
                },
            )
            events.add_error_object(
                log,
                observation_object_id,
                kind="ExecError" if exec_error else "NonZeroExit",
                message=message,
            )

        record_turn(
            TurnRecord(
                step=step,
                reasoning=reasoning,
                raw_response=result.text,
                command=command,
                done=done,
                parse_failed=False,
                exit_code=exit_code,
                feedback_text=feedback,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cache_creation_input_tokens=result.cache_creation_input_tokens,
                cache_read_input_tokens=result.cache_read_input_tokens,
                latency_sec=result.latency_sec,
                command_wall_time_sec=command_wall_time,
            )
        )
        step += 1
