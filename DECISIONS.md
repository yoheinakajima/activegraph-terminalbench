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

NOT verified here, and why:

- **No real Anthropic API call was made.** This sandbox has no
  `ANTHROPIC_API_KEY`. The haiku smoke test and the sonnet full run in the
  README are written exactly as they should be typed on your Mac, but they
  were not executed. The first thing to do after cloning is run smoke test
  2 and read one `event_log.json` end to end.
- **Nothing was executed on macOS or Apple Silicon.** All Docker
  verification here is linux/amd64. The Rosetta guidance in the README is
  standard TB2 practice but was not exercised in this build.
- **The full 89-task oracle run was not executed here** (sandbox network
  policy blocks several verifier downloads, notably the uv installer from
  astral.sh that most TB2 verifiers fetch; that restriction does not exist
  on a normal network).
- Sandbox-only adaptations that were used here and are NOT part of the
  repo: a local `registry.json` with `--registry-path` (the hub registry
  API was proxy-blocked; the README documents this as a troubleshooting
  fallback), a Docker registry mirror, and an `--extra-docker-compose`
  overlay mounting the proxy's CA bundle into task containers. None are
  needed on a machine with normal egress.

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
