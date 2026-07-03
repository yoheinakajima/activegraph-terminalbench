"""The retrieval seam: everything the model sees comes from build_context().

v1 is deliberately dumb: system prompt, task instruction, and a verbatim
tail of the recent transcript, all reconstructed from the trial's event
log. Nothing else in the codebase assembles model input.

The transcript is a pure projection of the ActiveGraph event store:
- model_turn events become assistant messages (the raw model response)
- output_observed / parse_error / noop_turn events become user messages
  (the feedback text that was fed back to the model at the time)

Because the context is rebuilt from the log every turn, swapping in v2
changes retrieval without touching the loop.
"""

from __future__ import annotations

from activegraph_harness import events
from activegraph_harness.events import TrialLog
from activegraph_harness.prompts import render_budget_line, render_system_prompt

# Keep at most this many trailing (assistant, user) exchange pairs verbatim.
MAX_TAIL_EXCHANGES = 20

OMISSION_MARKER = (
    "[transcript trimmed: {n} earlier exchanges omitted; "
    "the full history is in the trial event log]"
)


def build_context(log: TrialLog, instruction: str, step_count: int, budgets) -> list[dict]:
    """v1: system prompt + instruction + verbatim tail of recent steps.

    Args:
        log: The trial's TrialLog. The transcript is projected from its
            event store; v2 will query its graph instead.
        instruction: The task instruction.
        step_count: Zero-based index of the turn about to run.
        budgets: loop.Budgets; used only to render the budget status line.

    Returns:
        A message list for llm.LLMClient.complete(). messages[0] is the
        system message; the rest alternate user/assistant and end with
        a user message. Message content is a plain string or a list of
        text blocks; blocks flagged "cache": True are stable across turns
        and become cache breakpoints (see llm.apply_cache_control).

    v2 (future): replace the verbatim tail with a retrieved subgraph:
    the active error object and its `contains` chain, the files it
    relates to, and prior attempts on the current subgoal (steps linked
    by `follows` edges that issued similar commands). The contract is
    identical: same signature, same return shape, messages[0] is the
    system message, the last message has role "user". The loop must
    never know which version it is talking to.
    """
    system = {
        "role": "system",
        "content": render_system_prompt(
            command_timeout_sec=budgets.command_timeout_sec
        ),
    }

    exchanges = _project_exchanges(log)
    omitted = max(0, len(exchanges) - MAX_TAIL_EXCHANGES)
    tail = exchanges[omitted:]

    first_user = f"Task instruction:\n{instruction}"
    if omitted:
        first_user += "\n\n" + OMISSION_MARKER.format(n=omitted)

    messages: list[dict] = [system, {"role": "user", "content": first_user}]
    for assistant_text, user_text in tail:
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": user_text})

    # The moving cache breakpoint sits on the final feedback block ("cache":
    # True, translated by llm.apply_cache_control), so each turn reads the
    # whole prior transcript from cache and writes only the new tail. The
    # per-turn budget countdown rides in an unmarked block AFTER it and
    # never invalidates the cached prefix. Caching tradeoff, v1: a big
    # input that is mostly cache reads while the tail window is not yet
    # sliding. Once trimming starts, the omission counter in the first user
    # message changes every turn, so the prefix can never hit again; the
    # breakpoint is dropped then, because a guaranteed-miss write bills
    # 1.25x for cache entries nothing will ever read (measured live: every
    # post-trim turn rewrote ~11k tokens for zero reads). Post-trim turns
    # pay plain input price. That is a v1 structural cost; v2 exists to
    # remove it.
    final = messages[-1]
    feedback_block: dict = {"type": "text", "text": final["content"]}
    if not omitted:
        feedback_block["cache"] = True
    final["content"] = [
        feedback_block,
        {
            "type": "text",
            "text": render_budget_line(budgets.status_line(step_count)),
        },
    ]
    _log_context_built(log, version="v1", step=step_count, messages=messages)
    return messages


def build_context_v2(log: TrialLog, instruction: str, step_count: int, budgets) -> list[dict]:
    """v2: stable cached prefix + a small retrieved slice from the graph.

    Same contract as build_context() (same signature, same return shape,
    the loop never knows which version it got). The content differs:

    Stable prefix, cached (system prompt + the task instruction block,
    marked "cache": True): identical every turn of the trial.

    Retrieved slice, rebuilt each turn and uncached BY DESIGN: the active
    Error object with its linked Command and Observation, files touched so
    far, prior attempts on the current subgoal (commands sharing the failing
    command's program), a one-line-per-step digest of older history, and
    the last V2_TAIL_EXCHANGES exchanges verbatim. All of it except the
    verbatim tail comes from typed graph queries (objects/relations over
    task/step/command/observation/error/file and issued/produced/contains/
    follows/touches edges).

    Caching tradeoff: v1-cached pays a big input that is mostly cache
    reads; v2 pays a small input that is mostly uncached (only the prefix
    is cached, and caching the slice would bill 1.25x for entries the next
    turn's rebuilt slice could never hit). Both should be cheap; they get
    there differently. The run metrics decide which wins.
    """
    system = {
        "role": "system",
        "content": render_system_prompt(
            command_timeout_sec=budgets.command_timeout_sec
        ),
    }

    slice_text, included_ids = _v2_retrieved_slice(log)
    exchanges = _project_exchanges(log)
    tail = exchanges[max(0, len(exchanges) - V2_TAIL_EXCHANGES):]
    if tail:
        slice_text += f"\n\nThe last {len(tail)} exchange(s) follow verbatim."

    first_user_blocks = [
        {"type": "text", "text": f"Task instruction:\n{instruction}", "cache": True},
        {"type": "text", "text": slice_text},
    ]
    messages: list[dict] = [system, {"role": "user", "content": first_user_blocks}]
    for assistant_text, user_text in tail:
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": user_text})

    budget_line = render_budget_line(budgets.status_line(step_count))
    final = messages[-1]
    if final["role"] == "user" and isinstance(final["content"], str):
        final["content"] = f"{final['content']}\n\n{budget_line}"
    else:
        first_user_blocks.append({"type": "text", "text": budget_line})

    _log_context_built(
        log,
        version="v2",
        step=step_count,
        messages=messages,
        extra={"included_object_ids": included_ids},
    )
    return messages


# Verbatim tail kept by v2; everything older arrives only through the
# digest and graph retrieval.
V2_TAIL_EXCHANGES = 5

# Bounds on the retrieved slice so it stays small by construction.
V2_MAX_FILES = 30
V2_MAX_PRIOR_ATTEMPTS = 5
V2_MAX_DIGEST_LINE_CHARS = 140
V2_MAX_ERROR_CHARS = 1200
V2_MAX_PREVIEW_CHARS = 500


def _v2_retrieved_slice(log: TrialLog) -> tuple[str, dict]:
    """Assemble the retrieved slice through typed graph queries."""
    graph = log.graph
    included: dict = {}
    sections: list[str] = ["=== Retrieved context (from the run graph) ==="]

    steps = sorted(graph.objects(type="step"), key=lambda o: o.data["index"])
    step_by_command_id = {}
    command_by_step_id = {}
    for step in steps:
        for rel in graph.relations(source=step.id, type="issued"):
            command = graph.get_object(rel.target)
            if command is not None:
                step_by_command_id[command.id] = step
                command_by_step_id[step.id] = command

    def observation_for(command_id: str):
        for rel in graph.relations(source=command_id, type="produced"):
            obs = graph.get_object(rel.target)
            if obs is not None:
                return obs
        return None

    # Files touched so far.
    files = graph.objects(type="file")
    if files:
        shown = [f.data["path"] for f in files[-V2_MAX_FILES:]]
        omitted = len(files) - len(shown)
        suffix = f" (+{omitted} older)" if omitted else ""
        sections.append(f"Files touched so far: {', '.join(shown)}{suffix}")
        included["file_ids"] = [f.id for f in files[-V2_MAX_FILES:]]

    # The active error: the most recent error object, with its linked
    # observation -> command -> step chain.
    errors = graph.objects(type="error")
    active_program = None
    if errors:
        error = errors[-1]
        included["error_id"] = error.id
        lines = [
            f"Active error ({error.data['kind']}): "
            f"{error.data['message'][:V2_MAX_ERROR_CHARS]}"
        ]
        for rel in graph.relations(target=error.id, type="contains"):
            obs = graph.get_object(rel.source)
            if obs is None:
                continue
            included["observation_id"] = obs.id
            for produced in graph.relations(target=obs.id, type="produced"):
                command = graph.get_object(produced.source)
                if command is None:
                    continue
                included["command_id"] = command.id
                active_program = command.data["text"].split()[:1]
                step = step_by_command_id.get(command.id)
                where = f" (step {step.data['index']})" if step else ""
                lines.append(f"  command{where}: $ {command.data['text']}")
                lines.append(
                    f"  output preview: {obs.data['preview'][:V2_MAX_PREVIEW_CHARS]}"
                )
        sections.append("\n".join(lines))

    # Prior attempts on the current subgoal: earlier commands sharing the
    # failing command's program token.
    if active_program:
        program = active_program[0]
        attempts = []
        for command in graph.objects(type="command"):
            if command.id == included.get("command_id"):
                continue
            if command.data["text"].split()[:1] != active_program:
                continue
            obs = observation_for(command.id)
            step = step_by_command_id.get(command.id)
            index = step.data["index"] if step else "?"
            exit_code = obs.data["exit_code"] if obs else "?"
            attempts.append(
                f"  step {index}: $ {command.data['text'][:100]} (exit {exit_code})"
            )
        if attempts:
            shown = attempts[-V2_MAX_PRIOR_ATTEMPTS:]
            sections.append(
                f"Prior attempts with `{program}` ({len(attempts)}):\n"
                + "\n".join(shown)
            )
            included["n_prior_attempts"] = len(attempts)

    # One-line-per-step digest of everything older than the verbatim tail.
    digest_steps = steps[: max(0, len(steps) - V2_TAIL_EXCHANGES)]
    if digest_steps:
        lines = []
        for step in digest_steps:
            command = command_by_step_id.get(step.id)
            if command is None:
                lines.append(f"  step {step.data['index']}: (no command)")
                continue
            obs = observation_for(command.id)
            exit_code = obs.data["exit_code"] if obs else "?"
            line = f"  step {step.data['index']}: $ {command.data['text']} (exit {exit_code})"
            lines.append(line[:V2_MAX_DIGEST_LINE_CHARS])
        sections.append("History digest (older steps, one line each):\n" + "\n".join(lines))
        included["n_digest_steps"] = len(digest_steps)

    if len(sections) == 1:
        sections.append("(no history yet: this is the first step)")
    return "\n\n".join(sections), included


def _estimate_tokens(messages: list[dict]) -> int:
    chars = 0
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            chars += len(content)
        else:
            chars += sum(len(block.get("text", "")) for block in content)
    return chars // 4


def _log_context_built(
    log: TrialLog,
    *,
    version: str,
    step: int,
    messages: list[dict],
    extra: dict | None = None,
) -> None:
    """The audit trail for why the model saw what it saw this turn."""
    events.append_event(
        log,
        "context_built",
        step=step,
        payload={
            "version": version,
            "n_messages": len(messages),
            "token_estimate": _estimate_tokens(messages),
            **(extra or {}),
        },
    )


def _project_exchanges(log: TrialLog) -> list[tuple[str, str]]:
    """Project the event log into (assistant_text, user_feedback) pairs.

    Every model_turn is followed (possibly not immediately) by exactly one
    feedback event carrying the text that was fed back to the model:
    output_observed, parse_error, or noop_turn. Feedback events for the
    same step are joined if more than one exists.
    """
    pairs: list[tuple[str, list[str]]] = []
    for event in log.graph.events:
        if event.type == "model_turn":
            pairs.append((event.payload["raw_response"], []))
        elif event.type in ("output_observed", "parse_error", "noop_turn"):
            feedback = event.payload.get("feedback_text", "")
            if pairs and feedback:
                pairs[-1][1].append(feedback)
    return [
        (assistant, "\n\n".join(feedback) if feedback else "(no output)")
        for assistant, feedback in pairs
    ]
