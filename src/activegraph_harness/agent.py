"""ActiveGraphAgent: the Harbor-facing entry point.

An external agent: the loop, the LLM calls, and the ActiveGraph store all
run on the host, driving the task container through environment.exec().
Per trial it:

1. opens a fresh ActiveGraph store (sqlite file under self.logs_dir),
2. appends task_started and hands off to loop.run_loop(),
3. mirrors every finished turn into Harbor's AgentContext and an ATIF
   trajectory.json (dumped after every turn so a timeout still leaves a
   usable partial trajectory),
4. always exports the full event log JSON plus summary.json in finally.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import override

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import (
    Agent as AtifAgent,
    FinalMetrics,
    Metrics,
    Observation,
    ObservationResult,
    Step,
    ToolCall,
    Trajectory,
)
from harbor.utils.import_path import import_class
from harbor.utils.trajectory_utils import format_trajectory_json

from activegraph_harness import __version__, events
from activegraph_harness.context import build_context, build_context_v2
from activegraph_harness.events import TrialLog
from activegraph_harness.llm import LLMClient, resolve_model
from activegraph_harness.loop import (
    DEFAULT_COMMAND_TIMEOUT_SEC,
    DEFAULT_MAX_STEPS,
    DEFAULT_WALL_CLOCK_BUDGET_SEC,
    Budgets,
    LoopResult,
    TurnRecord,
    run_loop,
)

EVENT_LOG_FILENAME = "event_log.json"
SUMMARY_FILENAME = "summary.json"
STORE_FILENAME = "events.sqlite"
TRAJECTORY_FILENAME = "trajectory.json"

CONTEXT_BUILDERS = {"v1": build_context, "v2": build_context_v2}


def _parse_bool(value: object) -> bool:
    """Harbor passes --ak values as strings; accept real bools too."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
    raise ValueError(f"expected a boolean, got {value!r}")


class ActiveGraphAgent(BaseAgent):
    SUPPORTS_ATIF: bool = True

    def __init__(
        self,
        logs_dir: Path,
        model_name: str | None = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        command_timeout_sec: int = DEFAULT_COMMAND_TIMEOUT_SEC,
        wall_clock_budget_sec: float = DEFAULT_WALL_CLOCK_BUDGET_SEC,
        llm_client_import_path: str = "activegraph_harness.llm:LLMClient",
        enable_prompt_caching: bool = True,
        context_version: str = "v1",
        *args,
        **kwargs,
    ):
        """All knobs are Harbor agent kwargs (--ak key=value).

        Args:
            max_steps: Maximum ReAct turns per trial.
            command_timeout_sec: Per-command timeout inside the container.
            wall_clock_budget_sec: Soft wall clock budget for the whole loop.
            llm_client_import_path: 'module:Class' of the model client. The
                default is the Anthropic wrapper; tests substitute a
                scripted client here.
            enable_prompt_caching: Place cache_control breakpoints on
                requests (see llm.apply_cache_control). On by default;
                --ak enable_prompt_caching=false restores pass 1 behavior.
            context_version: Which build_context the loop gets: "v1"
                (verbatim transcript tail) or "v2" (graph retrieval).
                --ak context_version=v2 selects v2; the loop never knows.
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self._max_steps = int(max_steps)
        self._command_timeout_sec = int(command_timeout_sec)
        self._wall_clock_budget_sec = float(wall_clock_budget_sec)
        self._llm_client_import_path = llm_client_import_path
        self._enable_prompt_caching = _parse_bool(enable_prompt_caching)
        if context_version not in CONTEXT_BUILDERS:
            raise ValueError(
                f"unknown context_version {context_version!r}; "
                f"expected one of {sorted(CONTEXT_BUILDERS)}"
            )
        self._context_version = context_version
        self._environment_info: dict[str, str] = {}

    @staticmethod
    @override
    def name() -> str:
        return "activegraph"

    @override
    def version(self) -> str | None:
        return __version__

    @override
    async def setup(self, environment: BaseEnvironment) -> None:
        """External agent: nothing to install in the container. Capture the
        environment details here; they go into the task_started event once
        the per-trial store exists in run()."""
        self._environment_info = {
            "environment_type": str(environment.type()),
            "environment_name": environment.environment_name,
            "session_id": environment.session_id,
            "os": str(environment.os),
            "default_user": str(environment.default_user),
        }
        self.logger.info(f"setup complete (external agent): {self._environment_info}")

    @override
    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        model = resolve_model(self.model_name)
        trial_id = str(self.context_id) if self.context_id else str(uuid.uuid4())
        task_id = environment.environment_name

        log = events.open_trial_log(
            self.logs_dir / STORE_FILENAME,
            trial_id=trial_id,
            task_id=task_id,
            model=model,
        )
        recorder = _TrajectoryRecorder(
            logs_dir=self.logs_dir,
            context=context,
            instruction=instruction,
            model=model,
            agent_version=self.version() or "unknown",
        )
        budgets = Budgets(
            max_steps=self._max_steps,
            wall_clock_sec=self._wall_clock_budget_sec,
            command_timeout_sec=self._command_timeout_sec,
        )
        started = time.monotonic()
        loop_result: LoopResult | None = None
        fatal_error: str | None = None

        events.append_event(
            log,
            "task_started",
            step=0,
            payload={
                "instruction": instruction,
                "environment": self._environment_info,
                "budgets": {
                    "max_steps": budgets.max_steps,
                    "wall_clock_sec": budgets.wall_clock_sec,
                    "command_timeout_sec": budgets.command_timeout_sec,
                },
                "context_version": self._context_version,
                "prompt_caching": self._enable_prompt_caching,
            },
        )

        try:
            llm_class = import_class(self._llm_client_import_path, label="llm client")
            llm = llm_class(
                self.model_name, enable_prompt_caching=self._enable_prompt_caching
            )
            loop_result = await run_loop(
                environment=environment,
                instruction=instruction,
                log=log,
                llm=llm,
                budgets=budgets,
                record_turn=recorder.record,
                context_builder=CONTEXT_BUILDERS[self._context_version],
            )
        except Exception as exc:
            fatal_error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            wall_time = round(time.monotonic() - started, 3)
            n_events = 0
            try:
                n_events = events.export_log(log, self.logs_dir / EVENT_LOG_FILENAME)
                events.write_summary(
                    log,
                    self.logs_dir / SUMMARY_FILENAME,
                    {
                        "finish_reason": (
                            loop_result.finish_reason if loop_result else "fatal_error"
                        ),
                        "fatal_error": fatal_error,
                        "steps": recorder.n_turns,
                        "commands": recorder.n_commands,
                        "errors": loop_result.n_errors if loop_result else None,
                        "tokens_in": recorder.total_prompt_tokens(),
                        "tokens_in_uncached": recorder.total_input_tokens,
                        "tokens_out": recorder.total_output_tokens,
                        "tokens_cache_creation": recorder.total_cache_creation_tokens,
                        "tokens_cache_read": recorder.total_cache_read_tokens,
                        "cache_hit_rate": recorder.cache_hit_rate(),
                        "context_version": self._context_version,
                        "prompt_caching": self._enable_prompt_caching,
                        "wall_time_sec": wall_time,
                    },
                )
            finally:
                events.close_trial_log(log)
            context.metadata = {
                "finish_reason": (
                    loop_result.finish_reason if loop_result else "fatal_error"
                ),
                "fatal_error": fatal_error,
                "n_steps": recorder.n_turns,
                "n_commands": recorder.n_commands,
                "n_events": n_events,
                "wall_time_sec": wall_time,
                "event_log": EVENT_LOG_FILENAME,
                "event_store": STORE_FILENAME,
                "summary": SUMMARY_FILENAME,
            }
            recorder.dump()


class _TrajectoryRecorder:
    """Mirrors TurnRecords into Harbor's AgentContext and an ATIF
    trajectory.json, incrementally, so partial runs stay inspectable."""

    def __init__(
        self,
        *,
        logs_dir: Path,
        context: AgentContext,
        instruction: str,
        model: str,
        agent_version: str,
    ):
        self._logs_dir = logs_dir
        self._context = context
        self._model = model
        self._agent_version = agent_version
        self._session_id = str(uuid.uuid4())
        self.n_turns = 0
        self.n_commands = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cache_creation_tokens = 0
        self.total_cache_read_tokens = 0
        self._steps: list[Step] = [
            Step(
                step_id=1,
                timestamp=self._now(),
                source="user",
                message=instruction,
            )
        ]

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def total_prompt_tokens(self) -> int:
        """All prompt tokens the model saw. With caching on, the API's
        input_tokens counts only the uncached suffix after the last cache
        breakpoint; the full prompt is the sum of all three counters. This
        keeps tokens_in comparable with the uncached pass 1 numbers."""
        return (
            self.total_input_tokens
            + self.total_cache_creation_tokens
            + self.total_cache_read_tokens
        )

    def cache_hit_rate(self) -> float:
        """Share of all prompt tokens that were served from cache."""
        total = self.total_prompt_tokens()
        if total == 0:
            return 0.0
        return round(self.total_cache_read_tokens / total, 4)

    def record(self, turn: TurnRecord) -> None:
        self.n_turns += 1
        if turn.command is not None:
            self.n_commands += 1
        self.total_input_tokens += turn.input_tokens
        self.total_output_tokens += turn.output_tokens
        self.total_cache_creation_tokens += turn.cache_creation_input_tokens
        self.total_cache_read_tokens += turn.cache_read_input_tokens

        tool_calls: list[ToolCall] | None = None
        observation_results: list[ObservationResult] = []
        if turn.command is not None:
            call_id = f"call_{turn.step}_1"
            tool_calls = [
                ToolCall(
                    tool_call_id=call_id,
                    function_name="bash_command",
                    arguments={"command": turn.command},
                    extra={"wall_time_sec": turn.command_wall_time_sec},
                )
            ]
            observation_results.append(
                ObservationResult(
                    source_call_id=call_id,
                    content=turn.feedback_text,
                    extra={"exit_code": turn.exit_code},
                )
            )
        elif turn.feedback_text:
            observation_results.append(ObservationResult(content=turn.feedback_text))

        self._steps.append(
            Step(
                step_id=len(self._steps) + 1,
                timestamp=self._now(),
                source="agent",
                model_name=self._model,
                message=turn.reasoning or turn.raw_response,
                tool_calls=tool_calls,
                observation=(
                    Observation(results=observation_results)
                    if observation_results
                    else None
                ),
                metrics=Metrics(
                    prompt_tokens=turn.input_tokens
                    + turn.cache_creation_input_tokens
                    + turn.cache_read_input_tokens,
                    completion_tokens=turn.output_tokens,
                    cached_tokens=turn.cache_read_input_tokens,
                ),
                extra={
                    "parse_failed": turn.parse_failed,
                    "done": turn.done,
                    "latency_sec": turn.latency_sec,
                },
            )
        )

        self._context.n_input_tokens = self.total_prompt_tokens()
        self._context.n_cache_tokens = self.total_cache_read_tokens
        self._context.n_output_tokens = self.total_output_tokens
        self.dump()

    def dump(self) -> None:
        trajectory = Trajectory(
            session_id=self._session_id,
            agent=AtifAgent(
                name=ActiveGraphAgent.name(),
                version=self._agent_version,
                model_name=self._model,
            ),
            steps=self._steps,
            final_metrics=FinalMetrics(
                total_prompt_tokens=self.total_prompt_tokens(),
                total_completion_tokens=self.total_output_tokens,
                total_cached_tokens=self.total_cache_read_tokens,
            ),
        )
        path = self._logs_dir / TRAJECTORY_FILENAME
        path.write_text(format_trajectory_json(trajectory.to_json_dict()))
