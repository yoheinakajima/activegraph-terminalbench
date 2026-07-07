# activegraph-harness

A custom [Harbor](https://github.com/laude-institute/harbor) agent for Terminal-Bench 2.0 that uses [ActiveGraph](https://pypi.org/project/activegraph/) as its memory and logging substrate. The agent is a plain ReAct loop driving the task container through `environment.exec()`; the twist is that every step of every trial (model reasoning, command, output, error, timing, tokens) is appended as an event to a per-trial ActiveGraph store, so the run IS the event log and the typed graph (Task, Step, Command, Observation, Error) is its deterministic projection. Same base model, different harness, and every run is fully inspectable after the fact.

The one seam that matters: everything the model sees each turn comes from a single function, `build_context()` in `src/activegraph_harness/context.py`. Version 1 is deliberately dumb (system prompt plus the verbatim tail of the recent transcript, reconstructed from the event log), optionally with prompt caching (`--ak enable_cache=true`). Version 2 (`--ak context_version=v2`) swaps in graph-based retrieval behind that one function boundary: the active error object, files touched, prior attempts on the current subgoal, a one-line digest of older steps, and a short verbatim tail. The loop never knows which version it is talking to.

## Results so far

- **Pass 1** (Sonnet 4.6, k=1, all 89 tasks): [reports/pass1.md](reports/pass1.md). Headline: 38.2% canonical, 43.9% on the clean 82-task subset.
- **Pass 2** (Sonnet 4.5, k=3, three-config matrix: v1+cache, terminus-2 control, v2 retrieval): [reports/pass2.md](reports/pass2.md). Headline: v1+cache matches the control; v2 retrieval loses by 5 points, mechanism verified by instrumented reproduction (event stores in `results/events/`).
- **Write-up**: [blog/2026-07-terminal-bench-part-1.md](blog/2026-07-terminal-bench-part-1.md).
- **Pass 3 (planned)**: native x86 VM with open egress, uniform 3,600s timeouts, k=5, control on all 89 tasks, v3 hybrid context with graph-native stall detection.

## Prerequisites

- macOS on Apple Silicon.
- Docker Desktop installed and running, with **Settings, General, "Use Rosetta for x86_64/amd64 emulation on Apple Silicon"** enabled (some TB2 task images are amd64-only).
- Python 3.12+.
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) installed (`curl -LsSf https://astral.sh/uv/install.sh | sh`).
- An Anthropic API key exported: `export ANTHROPIC_API_KEY=sk-ant-...`

## Install

```bash
git clone https://github.com/yoheinakajima/activegraph-terminalbench.git
cd activegraph-terminalbench
uv sync
uv run harbor --help
```

If the last command prints Harbor's usage, you are ready.

## Smoke test 1: oracle (no LLM, no cost)

Proves Docker and Harbor work end to end by running Harbor's built-in oracle agent (it replays each task's reference solution, then the real verifier grades it):

```bash
uv run harbor run --dataset terminal-bench@2.0 --agent oracle --n-concurrent 4 --jobs-dir runs --job-name oracle-all
```

That runs all 89 tasks (roughly 1 to 2 hours on a laptop, image pulls dominate the first time). For a 5-minute version, run just two tasks:

```bash
uv run harbor run --dataset terminal-bench@2.0 --agent oracle -i largest-eigenval -i fix-code-vulnerability --n-concurrent 2 --jobs-dir runs --job-name oracle-smoke
```

Expect a mean reward near 1.0. (A handful of TB2 verifiers have drifting unpinned dependencies; an occasional oracle failure on a specific task, `build-cython-ext` at the time of writing, is a task problem, not a setup problem.)

## Smoke test 2: the agent on two easy tasks (cheap)

```bash
uv run harbor run \
  --dataset terminal-bench@2.0 \
  --agent activegraph_harness.agent:ActiveGraphAgent \
  --model anthropic/claude-haiku-4-5 \
  -i largest-eigenval -i fix-code-vulnerability \
  --n-concurrent 2 \
  --jobs-dir runs \
  --job-name agent-smoke
```

Expected runtime: 5 to 15 minutes including image pulls. Expected cost: well under $1 (two trials, tens of model turns, haiku pricing). Or run both smoke tests in one go with `bash scripts/smoke_test.sh`.

Useful agent knobs (all optional, passed as `--ak key=value`):

- `--ak max_steps=60`: maximum ReAct turns per trial.
- `--ak command_timeout_sec=180`: per-command timeout inside the container.
- `--ak wall_clock_budget_sec=840`: soft wall clock budget. The default matches the most common TB2 agent timeout (900s) minus headroom; raise it together with `--agent-timeout-multiplier` for long tasks.
- `--ak enable_cache=true`: prompt caching for the v1 transcript context (stable prefix breakpoints; the cache breakpoint is dropped once the transcript window starts sliding, see `context.py` for why).
- `--ak context_version=v2`: graph-retrieval context instead of the verbatim transcript.

## Full run (the expensive step)

WolfBench protocol: all 89 tasks, 5 trials each, sonnet:

```bash
uv run harbor run \
  --dataset terminal-bench@2.0 \
  --agent activegraph_harness.agent:ActiveGraphAgent \
  --model anthropic/claude-sonnet-4-6 \
  -k 5 \
  --n-concurrent 4 \
  --jobs-dir runs \
  --job-name wolfbench-sonnet
```

Be honest with yourself before pressing enter: this is 445 trials. On a single machine at `--n-concurrent 4`, with most tasks budgeted at 15 minutes and some at 1 to 3 hours, expect roughly **24 to 48 hours of wall time**. Cost depends heavily on how long the model persists per task; with sonnet pricing and up to 60 turns per trial, plan for a range of roughly **$150 to $500**, and watch the first few dozen trials before committing to the rest. Interrupting is safe: results land per trial as they finish (see Resuming below).

## Results

- Harbor writes one directory per trial under `runs/<job-name>/<task>__<id>/`, containing `result.json` (the pass/fail record the metrics script reads), `trial.log`, and `verifier/` output.
- The ActiveGraph artifacts land next to Harbor's own logs in each trial's `agent/` directory:
  - `events.sqlite`: the per-trial ActiveGraph store (durable event log).
  - `event_log.json`: the full ordered event list, exported after every trial.
  - `summary.json`: steps, commands, errors, tokens in/out, wall time, event counts.
  - `trajectory.json`: the same run in Harbor's ATIF format.

Compute the WolfBench metrics from one or more job directories:

```bash
uv run python scripts/wolfbench_metrics.py runs/wolfbench-sonnet --out-dir runs/wolfbench-sonnet
```

That writes `metrics.json` (solid, worst_of, average, best_of, ceiling, plus n_tasks, n_runs, and the solid-to-ceiling spread) and `matrix.csv` (tasks as rows, runs as columns, 0/1 cells, and an always/sometimes/never label per task). Reading the five numbers: solid is the floor you can trust (passes every time), ceiling is what is possible (passes at least once), and the spread between them is the variance meter; worst_of/average/best_of tell you how much a single run's headline number can flatter or slander the harness.

## Troubleshooting

- **Docker not running**: `harbor run` fails immediately with a docker compose error. Start Docker Desktop and rerun; nothing is left half-done.
- **amd64 images on Apple Silicon**: if a task fails with `no matching manifest for linux/arm64` or crashes instantly under emulation, confirm the Rosetta option in Docker Desktop is on. A few heavy amd64 images may still be flaky under emulation; document and exclude them with `-x <task-name>` rather than fighting them.
- **API rate limits**: 429s are retried 3 times with backoff (each retry is an `llm_retry` event in the trial's log). If they persist, lower `--n-concurrent` to 2 or 1.
- **Per-command timeouts**: commands that exceed `command_timeout_sec` come back to the model as an error observation with exit code -1; the model is told to background long-running work. Raise it with `--ak command_timeout_sec=300` for build-heavy tasks.
- **Resuming or rerunning a subset**: Harbor writes each trial as it finishes, so an interrupted job keeps everything already done. Rerun just the missing or failed tasks into a new job with `-i <task-name>` filters (repeatable, supports globs), then pass both job directories to `wolfbench_metrics.py`; it merges trials across directories per task.
- **Dataset or registry fetch fails**: download `registry.json` from the Harbor repo and pass `--registry-path /path/to/registry.json`, which skips the remote lookup.

## Repo map

```
src/activegraph_harness/
  agent.py     ActiveGraphAgent(BaseAgent): Harbor entry point, ATIF trajectory
  loop.py      the ReAct loop, model-agnostic, budgets, defensive parsing
  context.py   build_context() v1 + the documented v2 graph-retrieval seam
  events.py    ActiveGraph store, typed Pack schema, append/export helpers
  prompts.py   system prompt + response schema
  llm.py       Anthropic wrapper: retries, token accounting, timing
  testing.py   scripted model client for offline end-to-end runs
scripts/
  wolfbench_metrics.py   task-by-run matrix -> the five WolfBench numbers
  smoke_test.sh          oracle smoke + 2-task agent smoke
  run_config.sh          chunked matrix runner: per-chunk collect + commit + push
  collect_chunk.py       harbor job dir -> compact committed chunk artifact
  archive_events.sh      per-job event stores -> results/events/*.tar.gz (same commit)
  compare_runs.py        chunk artifacts -> five metrics, cost, per-task diffs
  recover_and_resume.sh  one-command recovery after an ephemeral-container reset
  rebuild_sandbox.sh     sandbox bootstrap (uv, registry snapshot, TLS overlays)
reports/                 pass1.md, pass2.md + committed metrics and matrices
results/                 per-chunk artifacts, event-store tarballs, exclusions
blog/                    the write-up series
```

See `DECISIONS.md` for every place this implementation diverges from the original spec and what was actually verified where.
