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
    return messages


def build_context_v2(log: TrialLog, instruction: str, step_count: int, budgets) -> list[dict]:
    """Stub for graph-based retrieval. See build_context() docstring for
    the contract. Swap the call site in loop.run_loop() when ready."""
    raise NotImplementedError("v2 graph retrieval is not implemented yet")


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
