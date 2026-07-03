"""System prompt and response schema for the ReAct loop.

The model must answer every turn with a single JSON object:

    {"reasoning": str, "command": str | null, "done": bool}

- reasoning: brief thinking about the current state and the next action.
- command: exactly one non-interactive shell command, or null for no action.
- done: true when the task is complete (command must be null in that case).
"""

from __future__ import annotations

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "reasoning": {"type": "string"},
        "command": {"type": ["string", "null"]},
        "done": {"type": "boolean"},
    },
    "required": ["reasoning", "command", "done"],
}

SYSTEM_PROMPT = """You are a terminal agent solving one task inside a Linux container.

Each turn you receive the transcript so far and must respond with a single JSON object and nothing else:

{{"reasoning": "<brief thinking>", "command": "<one shell command or null>", "done": <true or false>}}

Rules:
- Issue at most one shell command per turn. It runs via bash -c in the task container.
- Commands must be non-interactive. Never invoke editors, pagers, or anything that waits for input. Use flags like -y and tools like sed, awk, tee, printf, or heredocs to write files.
- Long output is truncated in the middle before it reaches you; a marker shows how much was cut. Full output is preserved in the run log.
- Each command has a {command_timeout_sec}s timeout. Prefer commands that finish quickly; background long-running processes and poll them.
- When you believe the task is complete, verify your work with a final check command first, then respond with "command": null and "done": true.
- A budget status line arrives with each turn. If it says you are nearly out of steps or time, wrap up: make the state as correct as possible, then finish.

The task instruction follows in the first user message."""

# Rendered fresh each turn and appended AFTER the last cache breakpoint, so
# it never invalidates the cached prefix. Moved out of the system prompt in
# pass 2: a per-turn countdown inside the system block would have broken the
# system cache on every request.
BUDGET_LINE = "Budget status: {budget_status}"


def render_system_prompt(*, command_timeout_sec: int) -> str:
    return SYSTEM_PROMPT.format(command_timeout_sec=command_timeout_sec)


def render_budget_line(budget_status: str) -> str:
    return BUDGET_LINE.format(budget_status=budget_status)
