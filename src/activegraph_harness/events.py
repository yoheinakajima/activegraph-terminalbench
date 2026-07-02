"""ActiveGraph store setup, event schema, append and export helpers.

One TrialLog per Harbor trial. The log is the agent: every step of the run
is appended as an event to a per-trial SQLite-backed ActiveGraph store, and
the typed graph (Task, Step, Command, Observation, Error objects plus their
edges) is the deterministic projection of those events.

Idioms used here, verified against activegraph 1.1.0 source:
- Runtime(graph, persist_to=path) is the documented happy path for opening
  a SQLiteEventStore and minting run metadata.
- Pack(ObjectType, RelationType) declares the typed schema; runtime.load_pack
  wires pydantic validation into graph.add_object / graph.add_relation.
- graph.emit(Event(...)) appends custom event types (task_started,
  model_turn, ...) to the log. Custom types are not projected onto the
  graph, which is exactly what we want for pure log entries.
- No behaviors are registered and the runtime queue is never drained, so
  loading the pack has no execution side effects.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from activegraph import Event, Graph, ObjectType, Pack, RelationType, Runtime
from pydantic import BaseModel

from activegraph_harness import __version__

# Every event appended by the harness carries this envelope inside its
# payload, in addition to the event-specific fields.
ENVELOPE_FIELDS = ("step", "trial_id", "task_id", "agent_version", "model")

# Append-only event vocabulary. "llm_retry" and "noop_turn" are harness
# extensions beyond the original spec list; see DECISIONS.md.
EVENT_TYPES = (
    "task_started",
    "model_turn",
    "command_executed",
    "output_observed",
    "error_raised",
    "parse_error",
    "noop_turn",
    "llm_retry",
    "budget_exhausted",
    "task_finished",
)


class TaskData(BaseModel):
    task_id: str
    instruction: str


class StepData(BaseModel):
    index: int
    reasoning: str


class CommandData(BaseModel):
    text: str
    timeout_sec: int


class ObservationData(BaseModel):
    exit_code: int
    stdout_chars: int
    stderr_chars: int
    wall_time_sec: float
    truncated: bool
    preview: str


class ErrorData(BaseModel):
    kind: str
    message: str


HARNESS_PACK = Pack(
    name="activegraph_harness",
    version=__version__,
    description="Typed schema for Terminal-Bench trial logs",
    object_types=(
        ObjectType(name="task", schema=TaskData, description="One benchmark task"),
        ObjectType(name="step", schema=StepData, description="One ReAct turn"),
        ObjectType(name="command", schema=CommandData, description="One shell command"),
        ObjectType(
            name="observation",
            schema=ObservationData,
            description="Result of executing a command",
        ),
        ObjectType(name="error", schema=ErrorData, description="An observed failure"),
    ),
    relation_types=(
        RelationType(name="issued", source_types=("step",), target_types=("command",)),
        RelationType(
            name="produced", source_types=("command",), target_types=("observation",)
        ),
        RelationType(
            name="contains", source_types=("observation",), target_types=("error",)
        ),
        RelationType(name="follows", source_types=("step",), target_types=("step",)),
    ),
)


@dataclass
class TrialLog:
    """Handle for one trial's event store plus the identity envelope."""

    graph: Graph
    runtime: Runtime
    db_path: Path
    trial_id: str
    task_id: str
    model: str


def open_trial_log(
    db_path: Path,
    *,
    trial_id: str,
    task_id: str,
    model: str,
) -> TrialLog:
    """Open a fresh per-trial store and load the typed schema pack."""
    graph = Graph()
    runtime = Runtime(graph, persist_to=str(db_path))
    runtime.load_pack(HARNESS_PACK)
    return TrialLog(
        graph=graph,
        runtime=runtime,
        db_path=db_path,
        trial_id=trial_id,
        task_id=task_id,
        model=model,
    )


def close_trial_log(log: TrialLog) -> None:
    store = log.graph.store
    if store is not None:
        store.close()


def append_event(
    log: TrialLog, event_type: str, *, step: int, payload: dict[str, Any]
) -> Event:
    """Append one harness event with the common envelope merged in."""
    if event_type not in EVENT_TYPES:
        raise ValueError(f"unknown event type {event_type!r}; add it to EVENT_TYPES")
    envelope = {
        "step": step,
        "trial_id": log.trial_id,
        "task_id": log.task_id,
        "agent_version": __version__,
        "model": log.model,
    }
    event = Event(
        id=log.graph.ids.event(),
        type=event_type,
        payload={**envelope, **payload},
        actor="agent",
        timestamp=log.graph.clock.now(),
    )
    log.graph.emit(event)
    return event


# ---------- graph projection helpers (typed objects and edges) ----------


def add_task_object(log: TrialLog, instruction: str) -> str:
    obj = log.graph.add_object(
        "task", {"task_id": log.task_id, "instruction": instruction}, actor="agent"
    )
    return obj.id


def add_step_object(
    log: TrialLog, index: int, reasoning: str, previous_step_id: str | None
) -> str:
    obj = log.graph.add_object(
        "step", {"index": index, "reasoning": reasoning}, actor="agent"
    )
    if previous_step_id is not None:
        log.graph.add_relation(previous_step_id, obj.id, "follows", actor="agent")
    return obj.id


def add_command_object(
    log: TrialLog, step_object_id: str, text: str, timeout_sec: int
) -> str:
    obj = log.graph.add_object(
        "command", {"text": text, "timeout_sec": timeout_sec}, actor="agent"
    )
    log.graph.add_relation(step_object_id, obj.id, "issued", actor="agent")
    return obj.id


def add_observation_object(
    log: TrialLog,
    command_object_id: str,
    *,
    exit_code: int,
    stdout: str,
    stderr: str,
    wall_time_sec: float,
    truncated: bool,
    preview: str,
) -> str:
    obj = log.graph.add_object(
        "observation",
        {
            "exit_code": exit_code,
            "stdout_chars": len(stdout),
            "stderr_chars": len(stderr),
            "wall_time_sec": wall_time_sec,
            "truncated": truncated,
            "preview": preview,
        },
        actor="agent",
    )
    log.graph.add_relation(command_object_id, obj.id, "produced", actor="agent")
    return obj.id


def add_error_object(
    log: TrialLog, observation_object_id: str | None, *, kind: str, message: str
) -> str:
    obj = log.graph.add_object(
        "error", {"kind": kind, "message": message}, actor="agent"
    )
    if observation_object_id is not None:
        log.graph.add_relation(
            observation_object_id, obj.id, "contains", actor="agent"
        )
    return obj.id


# ---------- export ----------


def export_log(log: TrialLog, path: Path) -> int:
    """Write the full ordered event list as one JSON file. Returns count.

    Reads from the durable store rather than the in-memory graph so the
    export proves what actually got persisted.
    """
    store = log.graph.store
    if store is None:
        raise RuntimeError("trial log has no attached store; nothing to export")
    events = [event.to_dict() for event in store.iter_events()]
    document = {
        "trial_id": log.trial_id,
        "task_id": log.task_id,
        "run_id": log.graph.run_id,
        "agent_version": __version__,
        "model": log.model,
        "n_events": len(events),
        "events": events,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False))
    return len(events)


def event_type_counts(log: TrialLog) -> dict[str, int]:
    counts: dict[str, int] = {}
    for event in log.graph.events:
        counts[event.type] = counts.get(event.type, 0) + 1
    return dict(sorted(counts.items()))


def write_summary(log: TrialLog, path: Path, summary: dict[str, Any]) -> None:
    """Write summary.json next to the event log. Caller supplies the totals;
    this adds identity and event counts so the file stands alone."""
    document = {
        "trial_id": log.trial_id,
        "task_id": log.task_id,
        "run_id": log.graph.run_id,
        "agent_version": __version__,
        "model": log.model,
        "event_counts": event_type_counts(log),
        **summary,
    }
    path.write_text(json.dumps(document, indent=2, ensure_ascii=False))
