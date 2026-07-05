# Pass 2: caching, a terminus-2 control, and the v2 retrieval treatment

Pass 2 ran the matrix that pass 1 motivated (see `reports/pass1.md`,
"What pass 2 changes and why"): the same activegraph harness on
Terminal-Bench 2.0 with Sonnet 4.5, k=3 trials per task, in three
configurations:

| config | agent | what it tests |
|---|---|---|
| A | activegraph v1 + prompt caching (`enable_cache=true`) | the pass-1 baseline made affordable: transcript-resend loop, but with cache reads at $0.30/MTok |
| B | terminus-2 | same-model, same-infra control, anchored to the published 42.8 ± 2.8 |
| C | activegraph v2 (`context_version=v2`) | the treatment: graph-retrieval `build_context` — small retrieved context instead of a big resent transcript |

All numbers computed by `scripts/compare_runs.py` from the committed chunk
artifacts in `results/pass2/` (metrics: `reports/pass2_metrics.json`,
matrices and per-task diffs: `reports/pass2_artifacts/`).

## Scope and basis

- **A and C: 86 of 89 tasks, k=3** (258–259 trials each). Three tasks were
  never attempted: `hf-model-inference`, `mteb-leaderboard`,
  `mteb-retrieve`. All three are in `results/excluded_tasks.json`
  (huggingface.co is blocked by the sandbox egress allowlist, proven in
  pass 1), so attempting them buys only guaranteed zeros — and the first
  one's Docker build cannot even complete behind the TLS-intercepting
  proxy; it crashed the runner twice before being dropped.
- **B: 24 tasks, k=3** (the first three chunks). B was stopped early by
  decision mid-run: terminus-2's uncached cost structure made the full
  86 tasks a poor use of budget once the control had enough tasks to
  anchor against (see cost section).
- **Clean subset** = tasks minus the 7 provably-unrunnable exclusions from
  pass 1, by construction from `results/excluded_tasks.json`: 82 tasks for
  A/C, 22 for B.
- One task got a fourth A trial from a re-run chunk (`s03b`); the
  comparison caps every task at its first 3 trials (`--max-runs 3`) so one
  over-sampled task does not force failure-padding onto the rest.
- Five C trials died with `VerifierTimeoutError` (no reward recorded):
  2× `caffe-cifar-10`, `filter-js-from-html`, `mailman`,
  `model-extraction-relu-logits`. They are counted as failures.

## Headline numbers (clean subset, k=3)

Five framings per config — `solid` = passed all 3 trials, `average` = mean
per-trial pass rate, `best_of` = best single run, `ceiling` = passed at
least once:

| config | tasks | solid | average | best-of | ceiling | cost |
|---|---|---|---|---|---|---|
| A (v1 + cache) | 82 | **22.0%** | **33.7%** | 37.8% | 46.3% | $182.74 |
| B (terminus-2) | 22 | 27.3% | 37.9% | 45.5% | 54.5% | $225.22* |
| C (v2 retrieval) | 82 | 17.1% | 28.5% | 30.5% | 40.2% | $149.87 |

*B's cost is a lower bound (LiteLLM does not surface cache-write counts,
so the 1.25× write premium is unpriced) and covers only 24 tasks × 3.

B's 82-task numbers do not exist, so the honest three-way comparison is on
the 22 clean tasks all three configs ran:

| config | solid | average | best-of | ceiling |
|---|---|---|---|---|
| A | 27.3% | 40.9% | 45.5% | 59.1% |
| B | 27.3% | 37.9% | 45.5% | 54.5% |
| C | 13.6% | 25.8% | 27.3% | 36.4% |

## Findings

**1. The cached v1 baseline matches the terminus-2 control.** On the
22-task common subset A and B are statistically indistinguishable (identical
solid and best-of; A's average is 3 points higher, well within noise at
n=22). The pass-1 conclusion stands with a real control behind it: the
harness's remaining gap to published terminus-2 numbers (42.8 ± 2.8
average) is not agent-loop quality — B itself only scores 37.9% average on
this subset in this sandbox, suggesting the sandbox (egress limits,
tainted tasks beyond the 7 proven ones) taxes both agents roughly equally.

**2. Prompt caching did its job.** A re-ran the pass-1 loop at k=3 for
$182.74 total — pass 1 paid $109 for k=1. Cache hit rate was 34.8%, and
the per-solve token bill was 816k. Caching is what made this matrix
runnable at all.

**3. The v2 retrieval treatment underperforms: −5.1 points average, −4.9
points solid versus A on the identical 82-task basis.** The regression is
concentrated, not diffuse: five tasks that A solves in ≥2 of 3 trials go
to zero under C (`cobol-modernization`, `compile-compcert`, `mailman`,
`mcmc-sampling-stan`, `portfolio-optimization`), against a single opposite
flip (`query-optimize`). A plausible common thread — unverified at the
transcript level — is that these are long, stateful tasks where the agent
needs its own earlier actions verbatim, which is what the retrieved-context
window compresses away; confirming that requires reading the archived
trial transcripts in `runs/archives/`.
Full per-task table: `reports/pass2_artifacts/diff_C_vs_A.csv`.

**4. v2's token economy is real but doesn't survive contact with cache
pricing.** C spends 524k tokens per solve to A's 816k (36% less), and its
input stream is nearly cache-free by design (1.1% hit rate — there is no
big stable prefix to cache). But cached v1 tokens are cheap: in dollars
per solve the two are a wash (A $2.18, C $2.14). The v2 design premise —
"resending the transcript is the expensive part" — was written before
caching landed; caching removed most of that expense without giving up
the transcript.

**What this suggests for a pass 3:** the v2 seam is worth keeping, but as
a *hybrid* — retrieval for cross-session/graph knowledge layered on top of
the verbatim recent transcript, not instead of it. And any future config
comparison should treat cached-v1 as the cost baseline to beat, not
uncached-v1.

## Operational appendix

The run survived five container reclaims, one disk-exhaustion crash, and
one hung trial across ~2 days of wall clock. What made it recoverable:

- **Chunked queue with per-chunk commit+push** (`scripts/run_config.sh`):
  every finished chunk's artifact goes to origin immediately; a reclaim
  costs at most the in-flight chunk (~30–60 min).
- **One-command recovery** (`scripts/recover_and_resume.sh`): restore
  branch, rebuild sandbox, resume queue; idempotent because collected
  chunks are skipped.
- **Giant-task chunks are serialized** (`--n-concurrent 1`) after
  `pytorch-model-recovery` at concurrency 4 filled the disk (ENOSPC), and
  the pre-chunk cleanup now prunes the buildkit cache, which
  `docker image prune` does not touch.
- **Harbor's per-task timeout can fail to fire**: one
  `schemelike-metacircular-eval` trial hung 2+ hours past its 40-minute
  budget with a live container and a dead agent; killing the container
  did not unstick harbor's event loop — the runner tree had to be killed
  by PID and the incomplete trial dir removed before resuming.
- The flock-based runner lock plus pidfile checks replaced `pgrep`
  liveness tests, which false-positive on their own command line.
