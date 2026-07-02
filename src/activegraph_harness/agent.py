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
        """
        super().__init__(logs_dir, model_name, *args, **kwargs)
        self._max_steps = int(max_steps)
        self._command_timeout_sec = int(command_timeout_sec)
        self._wall_clock_budget_sec = float(wall_clock_budget_sec)
        self._llm_client_import_path = llm_client_import_path
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
            },
        )

        try:
            llm_class = import_class(self._llm_client_import_path, label="llm client")
            llm = llm_class(self.model_name)
            loop_result = await run_loop(
                environment=environment,
                instruction=instruction,
                log=log,
                llm=llm,
                budgets=budgets,
                record_turn=recorder.record,
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
                        "tokens_in": recorder.total_input_tokens,
                        "tokens_out": recorder.total_output_tokens,
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

    def record(self, turn: TurnRecord) -> None:
        self.n_turns += 1
        if turn.command is not None:
            self.n_commands += 1
        self.total_input_tokens += turn.input_tokens
        self.total_output_tokens += turn.output_tokens

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
                    prompt_tokens=turn.input_tokens,
                    completion_tokens=turn.output_tokens,
                ),
                extra={
                    "parse_failed": turn.parse_failed,
                    "done": turn.done,
                    "latency_sec": turn.latency_sec,
                },
            )
        )

        self._context.n_input_tokens = self.total_input_tokens
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
                total_prompt_tokens=self.total_input_tokens,
                total_completion_tokens=self.total_output_tokens,
            ),
        )
        path = self._logs_dir / TRAJECTORY_FILENAME
        path.write_text(format_trajectory_json(trajectory.to_json_dict()))
