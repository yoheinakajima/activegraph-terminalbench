"""End-to-end test of the loop against a fake environment and a scripted
model, verifying the ActiveGraph event log, the typed graph projection,
and the export files. No Docker, no network."""

import asyncio
import json

import pytest

from activegraph_harness import events
from activegraph_harness.llm import LLMResult
from activegraph_harness.loop import Budgets, TurnRecord, parse_response, run_loop


class FakeExecResult:
    def __init__(self, stdout, stderr, return_code):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


class FakeEnvironment:
    def __init__(self):
        self.commands = []

    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.commands.append(command)
        if command == "false":
            return FakeExecResult("", "boom", 1)
        return FakeExecResult(f"ran: {command}", "", 0)


class FakeLLM:
    model = "fake"

    def __init__(self, turns):
        self._turns = list(turns)

    async def complete(self, messages, *, log_event):
        text = self._turns.pop(0)
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


def make_log(tmp_path):
    return events.open_trial_log(
        tmp_path / "events.sqlite",
        trial_id="trial-1",
        task_id="fake-task",
        model="fake",
    )


def test_parse_response_accepts_fenced_json():
    parsed = parse_response(
        'Sure!\n```json\n{"reasoning": "r", "command": "ls", "done": false}\n```'
    )
    assert parsed["command"] == "ls"


def test_parse_response_rejects_missing_keys():
    with pytest.raises(Exception):
        parse_response('{"reasoning": "r"}')


def test_loop_happy_path(tmp_path):
    log = make_log(tmp_path)
    llm = FakeLLM(
        [
            json.dumps({"reasoning": "look", "command": "echo hi", "done": False}),
            json.dumps({"reasoning": "fail once", "command": "false", "done": False}),
            "not json at all",
            json.dumps({"reasoning": "finish", "command": None, "done": True}),
        ]
    )
    env = FakeEnvironment()
    turns: list[TurnRecord] = []
    result = asyncio.run(
        run_loop(
            environment=env,
            instruction="do the thing",
            log=log,
            llm=llm,
            budgets=Budgets(max_steps=10, wall_clock_sec=60, command_timeout_sec=5),
            record_turn=turns.append,
        )
    )

    assert result.finish_reason == "task_finished"
    assert result.n_steps == 4
    assert result.n_commands == 2
    assert result.n_errors == 2  # one non-zero exit, one parse error
    assert env.commands == ["echo hi", "false"]

    counts = events.event_type_counts(log)
    assert "task_started" not in counts  # the agent appends it, not the loop
    assert counts["model_turn"] == 4
    assert counts["command_executed"] == 2
    assert counts["output_observed"] == 2
    assert counts["parse_error"] == 1
    assert counts["error_raised"] == 1
    assert counts["task_finished"] == 1

    # typed graph projection
    objects = log.graph.all_objects()
    types = sorted(o.type for o in objects)
    assert types.count("step") == 4
    assert types.count("command") == 2
    assert types.count("observation") == 2
    assert types.count("error") == 2
    relations = log.graph.all_relations()
    rel_types = sorted(r.type for r in relations)
    assert rel_types.count("issued") == 2
    assert rel_types.count("produced") == 2
    assert rel_types.count("contains") == 1
    assert rel_types.count("follows") == 3

    # export from the durable store
    export_path = tmp_path / "event_log.json"
    n = events.export_log(log, export_path)
    document = json.loads(export_path.read_text())
    assert document["n_events"] == n
    assert document["trial_id"] == "trial-1"
    assert [e["type"] for e in document["events"]].count("model_turn") == 4

    summary_path = tmp_path / "summary.json"
    events.write_summary(log, summary_path, {"steps": result.n_steps})
    summary = json.loads(summary_path.read_text())
    assert summary["event_counts"]["model_turn"] == 4
    events.close_trial_log(log)


def test_loop_budget_exhaustion(tmp_path):
    log = make_log(tmp_path)
    llm = FakeLLM(
        [json.dumps({"reasoning": "r", "command": "echo x", "done": False})] * 5
    )
    result = asyncio.run(
        run_loop(
            environment=FakeEnvironment(),
            instruction="loop forever",
            log=log,
            llm=llm,
            budgets=Budgets(max_steps=2, wall_clock_sec=60, command_timeout_sec=5),
            record_turn=lambda t: None,
        )
    )
    assert result.finish_reason == "budget_exhausted"
    assert result.n_steps == 2
    counts = events.event_type_counts(log)
    assert counts["budget_exhausted"] == 1
    events.close_trial_log(log)


def test_double_parse_failure_becomes_noop(tmp_path):
    log = make_log(tmp_path)
    llm = FakeLLM(
        [
            "garbage one",
            "garbage two",
            json.dumps({"reasoning": "ok", "command": None, "done": True}),
        ]
    )
    result = asyncio.run(
        run_loop(
            environment=FakeEnvironment(),
            instruction="x",
            log=log,
            llm=llm,
            budgets=Budgets(max_steps=10, wall_clock_sec=60, command_timeout_sec=5),
            record_turn=lambda t: None,
        )
    )
    assert result.finish_reason == "task_finished"
    counts = events.event_type_counts(log)
    assert counts["parse_error"] == 1
    assert counts["noop_turn"] == 1
    events.close_trial_log(log)
