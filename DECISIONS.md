# DECISIONS

Every place this implementation diverges from the original spec, plus what
was actually verified in which environment. The build happened in a Linux
cloud sandbox behind a TLS-intercepting egress proxy, with no Anthropic API
key available; that constrains what "verified" means below, and each case is
called out.

## Interfaces: where the real APIs won over the prompt's sketch

1. **`rollout_details` is not ATIF.** The spec said to populate
   `context.rollout_details` with the trajectory in ATIF format. In harbor
   0.16.1, `RolloutDetail` is a TypedDict of token IDs and logprobs for RL
   training, which the Anthropic Messages API does not return; ATIF is a
   separate thing, a `trajectory.json` written to the agent's logs dir
   (that is how terminus_2 does it, and `SUPPORTS_ATIF` gates trajectory
   conversion in `harbor.utils.traces_utils`). This agent follows the real
   interface: `SUPPORTS_ATIF = True`, an ATIF `trajectory.json` dumped
   after every turn, token totals and metadata on the `AgentContext`, and
   `rollout_details` left as None.

2. **`build_context(store, ...)` takes a `TrialLog`, not a bare store.**
   ActiveGraph's `EventStore` is a deliberately minimal append/iterate
   protocol; the graph projection and the identity envelope live on the
   `Graph`/`Runtime` pair. `TrialLog` bundles them. v1 projects the
   transcript from the event list; v2 will query the typed graph on the
   same handle. Same seam, richer handle.

3. **Store idiom: `Runtime(graph, persist_to=...)` plus `load_pack`.**
   Constructing `SQLiteEventStore` by hand is documented in the
   activegraph source as a low-frequency operator path; the happy path is
   the Runtime constructor, which opens the store, mints the run id, and
   writes the runs-table row. The harness registers no behaviors and never
   drains the runtime queue, so loading the pack has zero execution side
   effects; it only wires pydantic validation into `add_object` /
   `add_relation`. Consequence: the event log also contains the library's
   own `pack.loaded`, `object.created`, and `relation.created` events
   alongside the harness vocabulary. That is a feature (full provenance),
   not an accident.

4. **Agent timeout is invisible to the agent.** Harbor enforces the task's
   `[agent] timeout_sec` outside the agent (`asyncio.wait_for` in the
   trial), and `run()` receives no timeout parameter, so "a soft wall-clock
   budget derived from the task timeout" cannot be read at runtime. The
   budget is an agent kwarg (`--ak wall_clock_budget_sec=...`) defaulting
   to 840s: the modal TB2 agent timeout (900s, 49 of 89 tasks) minus 60s of
   headroom for export and shutdown. For the 3600s+ tasks you must raise it
   explicitly.

5. **The unified `--agent` flag takes the import path directly.** The spec
   mentioned `--agent-import-path`; current harbor accepts
   `--agent activegraph_harness.agent:ActiveGraphAgent`. README uses the
   real flag.

## Spec deviations by choice

6. **Repo naming.** The spec asked for a repo called `activegraph-harness`.
   The GitHub repo this was built in is `yoheinakajima/activegraph-terminalbench`,
   so the project lives at its root and the Python distribution/package is
   named `activegraph-harness` / `activegraph_harness`.

7. **Two extra event types.** `llm_retry` (each backoff retry, logged from
   inside the LLM wrapper) and `noop_turn` (model returned no command and
   not done, or two consecutive parse failures). Both fit the append-only
   vocabulary and keep the "log everything" rule honest.

8. **Setup events fold into `task_started`.** The per-trial store opens in
   `run()` (that is where `logs_dir` semantics and the trial identity are
   final), so `setup()` cannot append events yet. It captures the
   environment details instead, and they ship in the `task_started`
   payload.

9. **Observation storage split.** Full stdout/stderr goes in the
   `output_observed` event (the spec's "full output stored in the event").
   The typed `observation` graph object stores sizes, exit code, timing,
   and a 500-char preview rather than duplicating the full text; the event
   log remains the source of truth and the object stays cheap to load for
   v2 retrieval.

10. **Pass definition in metrics.** A trial passes iff its resolved reward
    is 1.0 (TB2 rewards are 0/1). Trials with no reward at all (exception,
    agent timeout before verification) count as failures and are listed
    under `incomplete_trials` in `metrics.json` instead of being dropped
    silently. If a future task emits multiple reward keys without a
    canonical `reward`, the script falls back to their mean.

11. **worst_of / best_of interpretation.** With solid defined as pass-in-all
    and ceiling as pass-in-any, worst_of and best_of are the per-run pass
    rates of the worst and best single run. That gives five distinct,
    monotonically ordered numbers: solid <= worst_of <= average <= best_of
    <= ceiling.

12. **Pluggable model client.** The agent takes
    `--ak llm_client_import_path=module:Class` (default: the Anthropic
    wrapper). This exists so the whole Harbor + Docker + ActiveGraph path
    can be exercised with a scripted, deterministic client
    (`activegraph_harness.testing:ScriptedLLM`) where no API key exists.
    It is also the reason the loop is genuinely model-agnostic.

13. **SDK retries disabled.** `anthropic.AsyncAnthropic(max_retries=0)`;
    the wrapper does its own backoff so every retry is an `llm_retry`
    event. Found the hard way: the SDK silently retried a 529 in testing,
    and 529 `OverloadedError` is not a subclass of `InternalServerError`,
    so the retry predicate is status-code based (429 and >= 500).

14. **Sampling parameters.** Temperature is left at the API default and
    `max_tokens` is 4096. Nothing in the spec asked otherwise; noted so
    nobody assumes temperature 0.

## What was verified where (be honest)

Verified in this sandbox, end to end:

- `uv sync` clean on a fresh checkout; `uv run harbor --help` works.
- Oracle runs on real TB2 tasks via Docker: `largest-eigenval` and
  `fix-code-vulnerability` both pass with reward 1.0 (also ran
  `build-cython-ext`, which fails one verifier test due to unpinned
  dependency drift inside the task, and `headless-terminal`, which fails
  under the sandbox's network restrictions; neither is a harness bug).
- The ActiveGraph agent completed `largest-eigenval` end to end through
  Harbor against the real container and real verifier, reward 1.0, using
  the scripted model client. `events.sqlite`, `event_log.json` (42 events,
  full vocabulary), `summary.json`, and an ATIF-valid `trajectory.json`
  all produced and hand-checked.
- A `-k 2` run over the two tasks (4 trials, all reward 1.0 per Harbor's
  own result.json files) fed `wolfbench_metrics.py`; the matrix matches
  Harbor's rewards cell for cell. The mixed-variance paths (sometimes /
  never labels, missing rewards, distinct five numbers) are covered by a
  synthetic-fixture test in `tests/test_metrics.py` because the live runs
  all passed.
- `LLMClient` retry, token accounting, and model-name resolution against a
  local mock Anthropic endpoint (`tests/test_llm.py`).
- The loop's parse-retry, no-op, budget-exhaustion, error-object, and
  export paths (`tests/test_loop.py`, against real activegraph, no mocks
  of the store).

Verified in a second sandbox session (2026-07-03), with a real API key
(`ANTHROP_API_KEY`):

- **Real Anthropic API calls are now verified.** Smoke test 2 ran against
  `claude-haiku-4-5` (2 tasks, then again with `-k 2`): per-turn
  `input_tokens` / `output_tokens` / `latency_sec` in `event_log.json` are
  real API usage numbers, `llm_retry` never fired (no 429s at
  4-concurrent), and `events.sqlite`, `event_log.json`, `summary.json`,
  `trajectory.json` were produced for every trial. `wolfbench_metrics.py`
  output was re-checked cell for cell against Harbor's result.json rewards,
  now including a live "sometimes" row (largest-eigenval flipped 1→0 on a
  wall-clock speedup assertion).
- **The full 89-task benchmark ran in this sandbox** with
  `anthropic/claude-sonnet-4-6`, k=1, `--n-concurrent 4`, in 4 chunks with
  `docker image prune` between them, plus one-at-a-time retries of 5 trials
  whose Docker environments never started (image-extract races with a
  concurrent prune, then ENOSPC on ~15GB torch/mteb images — sandbox disk,
  not the harness; 2 of the 5 passed on retry). Canonical result: 34/89 =
  38.2% pass (env errors counted as failures), 36/89 = 40.4% with the two
  recovered env-failures. ~$109 of sonnet spend (29.5M in / 1.4M out
  tokens), ~5h wall time. Per-task breakdown separating agent failures
  from sandbox-infrastructure failures: `results/README.md`.
- **The astral.sh blockage was worked around, and validated.** A local
  HTTPS stand-in for astral.sh serves an install.sh backed by the uv 0.9.5
  binaries repacked from the official PyPI wheels (gnu + musl), with a
  self-signed CA, `extra_hosts: astral.sh:host-gateway`, and a combined CA
  bundle in the compose overlay; a fully static curl is mounted at
  `/usr/local/bin/curl` because apt mirrors are also blocked and most
  verifiers `apt-get install curl` first (harmless failure — no `set -e` —
  once curl exists). Validated by the oracle agent on
  `openssl-selfsigned-cert`: reward 1.0. With this overlay every verifier
  except huggingface-dependent ones could run.

Still NOT verified, and why:

- **Nothing was executed on macOS or Apple Silicon.** All Docker
  verification here is linux/amd64. The Rosetta guidance in the README is
  standard TB2 practice but was not exercised in this build.
- **k>1 at full scale.** The full run is k=1 (a k=5 run would outlive this
  container), so solid/ceiling/spread coincide at 89 tasks; the variance
  machinery was demonstrated live only on the k=2 haiku smoke.
- **Tasks needing huggingface.co or Debian apt mirrors** never got a fair
  attempt (egress allowlist). They are counted as failures in the canonical
  number and itemized in `results/README.md`.
- Sandbox-only adaptations used here, NOT part of the repo and not needed
  on a machine with normal egress: local `registry.json` with
  `--registry-path`, a Docker registry mirror (`mirror.gcr.io`), and the
  `--extra-docker-compose` overlay described above (proxy CA bundle +
  astral.sh stand-in + static curl).

- **Phase-1 checkpoint decision (approved):** A and C complete the
  remaining 65 tasks at k=3; B (terminus-2 control) runs them at k=2.
  Rationale: phase-1 totals A 37.5% / B 34.7% / C 23.6% average; B cost
  >=$225 for 24 tasks (two 12000s build-pov-ray trials alone ~$160 via
  context-summarization loops), projecting the pass-2 total past the
  approved band at k=3. B keeps k=3 on the checkpoint-24 already run.
- **The sandbox reset twice more during pass 2** (disk snapshot rolled
  back to pass-1 state; second reset also wiped dockerd config, .cache,
  .venv). All chunk artifacts survived because the runner commits and
  pushes each chunk. Everything ephemeral is now reconstructed by
  scripts/rebuild_sandbox.sh (committed), validated after rebuild by a
  free oracle probe on openssl-selfsigned-cert (reward 1.0). The rebuilt
  stand-in matches the official uv installer contract (uv+uvx binaries,
  ~/.local/bin/env shim) and the overlays now export
  SSL_CERT_FILE/UV_NATIVE_TLS so uv/pip verify TLS through the sandbox
  MITM.

## Known fragilities

- **Anthropic-only.** `llm.py` speaks the Anthropic Messages API only, per
  spec. `--model` values with a non-anthropic provider prefix fail loudly.
- **Context growth.** v1 keeps the last 20 exchanges verbatim with 4k-char
  truncated observations; a very long trial's early context is dropped
  with a marker rather than summarized. That is the v2 seam's job.
- **`environment.exec` shells out to docker compose per command.** Fine
  for TB2's rates; would be slow for command-per-second workloads.
- **One store file per trial.** Chosen for isolation and because Harbor
  syncs the whole `agent/` dir; a shared cross-trial store (for cross-run
  memory experiments) would be a schema change, not a refactor, thanks to
  ActiveGraph's per-run scoping.

## Pass 2 decisions (2026-07-03, second session)

1. **Budget status moved out of the system prompt.** Pass 1 rendered the
   per-turn countdown inside the system block, which would have
   invalidated the system cache on every request. It now arrives as an
   uncached text block appended after the last cache breakpoint of the
   final message (v1) or inline after the retrieved slice (v2). This is
   the one model-visible divergence between pass 1 and Run A; the words
   are identical, only the position changed.
2. **Cache breakpoints are marked by the context builder,** as a
   harness-internal "cache": True flag on content blocks, translated to
   API cache_control by llm.apply_cache_control (which also always marks
   the system prompt, and rejects more than 3 marked blocks since the API
   caps breakpoints at 4). Placement is a retrieval concern: v1 marks the
   final feedback block (read the whole transcript from cache, write only
   the tail), v2 marks only the task instruction (the slice is rebuilt
   every turn, and caching it would bill 1.25x for entries that can never
   hit).
3. **v1 stops paying for cache once its window slides.** Measured live
   before the fix: after MAX_TAIL_EXCHANGES the omission counter changes
   every turn, so every request rewrote ~11k cache tokens for zero reads.
   Post-trim turns now carry no moving breakpoint and pay plain input
   price. Smoke test on fix-code-vulnerability (haiku, 39 turns): 53.2%
   cache hit rate, 0.53x effective input cost, no wasted writes.
4. **The smoke assertion was adapted from the spec.** "cache reads from
   turn 3 onward" holds only while the v1 window is stable; the committed
   check (scripts/cache_smoke_check.py) asserts reads for turns 3 through
   MAX_TAIL_EXCHANGES+1, near-zero cache writes after that, and effective
   cost below uncached.
5. **Schema additions for v2 (pack 0.2.0):** a `file` object type
   (path) and a `touches` relation (command -> file). Paths are extracted
   from command text by a deliberately loose heuristic
   (events.extract_file_paths); a false positive costs one stale context
   line. Pass 1 had no file tracking, and the v2 contract needs "files
   touched so far". Also added the `context_built` event type: version,
   message count, token estimate, and (v2) the retrieved object ids, per
   turn. Emitted by the builders themselves so the loop stays ignorant of
   the context version.
6. **v2 is selected by agent kwarg** (--ak context_version=v2), mapped to
   a context_builder callable passed into run_loop. The loop signature
   grew one defaulted parameter; the loop body still calls whatever
   builder it was handed.
7. **Run-matrix infrastructure incidents, all documented as they
   happened:** (a) the first Run A launch used pass-1-sized 24-task
   chunks; harbor schedules k=3 breadth-first, so all 24 task images plus
   ~11 GB of concurrent container writable layers must coexist, which
   cannot fit this disk. Killed at $21.90 of spend (36 agent-side trials,
   discarded for matrix cleanliness), rerun as 8-task chunks with the four
   giant-image tasks as singletons (scripts/run_config.sh, chunk files
   committed under results/pass2/chunks/). (b) One trial
   (custom-memory-heap-crash, Run A s03) hung forever in harbor's
   artifact-collection step after the agent finished: harbor idle at 0%
   CPU, verifier never started, no timeout applies there. Killed and
   replaced by a k=1 make-up trial of the same task (which passed); the
   swap is recorded inside results/pass2/pass2-A-s03.json. (c) terminus-2
   (Run B) installs tmux via apt or a source build, both egress-blocked,
   so every trial died at setup. Fixed by mounting a static tmux 3.3a
   (fetched from GitHub releases, reachable from containers) into B's
   containers only (tls-overlay-astral-tmux.yaml); with tmux present its
   installer no-ops. Verified with a one-task probe: reward 1.0, 82%
   cache hit rate.
