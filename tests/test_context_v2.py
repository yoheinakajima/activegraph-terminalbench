"""v2 graph-retrieval context: the retrieved slice must come from typed
graph queries, and both context versions must leave a context_built audit
trail. Runs the real loop against a fake environment and scripted model."""

import asyncio
import json

from activegraph_harness import events
from activegraph_harness.context import build_context, build_context_v2
from activegraph_harness.llm import LLMResult
from activegraph_harness.loop import Budgets, run_loop


class FakeExecResult:
    def __init__(self, stdout, stderr, return_code):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


class FakeEnvironment:
    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        if command.startswith("make"):
            return FakeExecResult("", "make: *** [all] Error 2", 2)
        return FakeExecResult(f"ran: {command}", "", 0)


class RecordingLLM:
    """Scripted model that keeps every message list it was shown."""

    model = "fake"

    def __init__(self, turns):
        self._turns = list(turns)
        self.seen: list[list[dict]] = []

    async def complete(self, messages, *, log_event):
        self.seen.append(messages)
        text = json.dumps(self._turns.pop(0))
        return LLMResult(
            text=text,
            input_tokens=10,
            output_tokens=5,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            latency_sec=0.01,
            model=self.model,
            retries=0,
            stop_reason="end_turn",
        )


SCRIPT = [
    {"reasoning": "look around", "command": "cat src/main.py", "done": False},
    {"reasoning": "build", "command": "make all", "done": False},
    {"reasoning": "retry build", "command": "make -j2 all", "done": False},
    {"reasoning": "check", "command": "ls docs/readme.md", "done": False},
    {"reasoning": "again", "command": "echo one", "done": False},
    {"reasoning": "again", "command": "echo two", "done": False},
    {"reasoning": "again", "command": "echo three", "done": False},
    {"reasoning": "finished", "command": None, "done": True},
]


def run(tmp_path, builder):
    log = events.open_trial_log(
        tmp_path / "events.sqlite", trial_id="t", task_id="task", model="fake"
    )
    llm = RecordingLLM(SCRIPT)
    result = asyncio.run(
        run_loop(
            environment=FakeEnvironment(),
            instruction="fix the build",
            log=log,
            llm=llm,
            budgets=Budgets(max_steps=20, wall_clock_sec=60),
            record_turn=lambda turn: None,
            context_builder=builder,
        )
    )
    return log, llm, result


def flatten(messages):
    parts = []
    for message in messages:
        content = message["content"]
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(block.get("text", "") for block in content)
    return "\n".join(parts)


def test_v2_slice_is_graph_retrieval(tmp_path):
    log, llm, result = run(tmp_path, build_context_v2)
    assert result.finish_reason == "task_finished"

    last = llm.seen[-1]
    text = flatten(last)

    # Files touched: extracted from commands, stored as file objects.
    file_paths = {o.data["path"] for o in log.graph.objects(type="file")}
    assert "src/main.py" in file_paths
    assert "docs/readme.md" in file_paths
    assert "Files touched so far" in text and "src/main.py" in text

    # Active error links back through contains -> produced -> issued.
    assert "Active error (NonZeroExit)" in text
    assert "make" in text

    # Prior attempts on the same program.
    assert "Prior attempts with `make`" in text

    # Older history arrives as a one-line digest, not verbatim.
    assert "History digest" in text

    # The stable prefix block is marked for caching; the slice is not.
    first_user = last[1]["content"]
    assert first_user[0]["cache"] is True
    assert "Task instruction" in first_user[0]["text"]
    assert "cache" not in first_user[1]

    # The audit trail names what was retrieved.
    built = [e for e in log.graph.events if e.type == "context_built"]
    assert len(built) == len(SCRIPT)
    payload = built[-1].payload
    assert payload["version"] == "v2"
    assert payload["token_estimate"] > 0
    ids = payload["included_object_ids"]
    assert ids["error_id"] in {o.id for o in log.graph.objects(type="error")}
    assert ids["file_ids"]


def test_v1_also_logs_context_built(tmp_path):
    log, llm, result = run(tmp_path, build_context)
    assert result.finish_reason == "task_finished"
    built = [e for e in log.graph.events if e.type == "context_built"]
    assert len(built) == len(SCRIPT)
    assert built[-1].payload["version"] == "v1"
    # v1 keeps the verbatim tail; no graph slice in what the model saw.
    assert "History digest" not in flatten(llm.seen[-1])
