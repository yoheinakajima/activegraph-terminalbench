# WolfBench results: activegraph-harness on Terminal-Bench 2.0

Full-benchmark attempt run on 2026-07-03 inside a Claude Code cloud sandbox.

- **Agent**: `activegraph_harness.agent:ActiveGraphAgent` (this repo, commit as of run)
- **Model**: `anthropic/claude-sonnet-4-6`, k=1, `--n-concurrent 4`
- **Dataset**: `terminal-bench@2.0` (all 89 tasks), run in 4 chunks
  (`runs/wolfbench-sonnet-k1-c1..c4`) plus sequential retries of trials whose
  Docker environment never started (`runs/wolfbench-sonnet-k1-retry*`).
- **API spend**: $105.96 sonnet (87 trials, 28.5M in / 1.36M out tokens) +
  $2.36 haiku smoke tests = **~$108 total** (estimated from public per-token
  pricing; the API does not return cost).

## Headline numbers (metrics.json / matrix.csv)

Canonical k=1 view over the four chunk jobs; trials that errored before the
verifier produced a reward count as failures:

| metric | value |
|---|---|
| pass rate (all five WolfBench numbers coincide at k=1) | **34/89 = 38.2%** |

With the five environment-startup failures retried (see below), the
infra-recovered pass rate is **36/89 = 40.4%** — and 40/89 = 44.9% if you
additionally exclude only-provably-sandbox-blocked tasks from the denominator.
`metrics.json` and `matrix.csv` keep the canonical 38.2% view; the retry
outcomes live in `runs/wolfbench-sonnet-k1-retry*` and are summarized here.

## Passing tasks (34)

bn-fit-modify, break-filter-js-from-html, cobol-modernization,
configure-git-webserver, constraints-scheduling, crack-7z-hash,
distribution-search, extract-elf, fix-code-vulnerability, fix-ocaml-gc,
git-leak-recovery, git-multibranch, install-windows-3.11, kv-store-grpc,
large-scale-text-editing, largest-eigenval, llm-inference-batching-scheduler,
mailman, modernize-scientific-stack, multi-source-data-merger,
openssl-selfsigned-cert, overfull-hbox, portfolio-optimization, pypi-server,
query-optimize, rstan-to-pystan, sanitize-git-repo,
schemelike-metacircular-eval, sqlite-db-truncate, torch-tensor-parallelism,
tune-mjcf, vulnerable-secret, write-compressor

## Sandbox-infrastructure failures (not agent failures)

The sandbox routes all egress through an allowlisting proxy. PyPI, npm,
crates.io and GitHub (from containers) work; Debian/Ubuntu apt mirrors,
huggingface.co, astral.sh and most other hosts are blocked. Workarounds used
for the whole run (documented in DECISIONS.md): a local astral.sh stand-in
serving the real uv 0.9.5 binaries from PyPI wheels, a static curl mounted at
`/usr/local/bin/curl`, and a combined CA bundle for the TLS-intercepting
proxy. Without these, ~82/89 verifiers cannot even start.

### Environment never started (Docker image pull/extract failures), retried sequentially

These five trials errored before the agent ran (image extract races while a
disk-space guard pruned concurrently, then ENOSPC on the ~15GB torch/mteb
images). Each was retried once, alone, on a clean disk — the agent's first
real attempt:

| task | retry outcome |
|---|---|
| custom-memory-heap-crash | **1.0 (pass)** |
| pytorch-model-recovery | **1.0 (pass)** |
| hf-model-inference | 0.0 — huggingface.co blocked (3/4 verifier tests pass; only `test_model_downloaded` fails) |
| mteb-retrieve | 0.0 — huggingface.co blocked (17 denied requests in agent transcript) |
| mteb-leaderboard | RETRY4_PLACEHOLDER |

### Blocked-egress failures during the agent's work

| task | evidence |
|---|---|
| count-dataset-tokens | agent's huggingface.co requests denied (`host_not_allowed`) |
| reshard-c4-data | verifier itself needs allenai/c4 from huggingface (errors in 4.7s) |
| build-pmars | verifier asserts Debian *source* packages were used; `apt-get source` impossible with mirrors blocked |
| make-doom-for-mips | agent blocked downloading gcc-12 MIPS cross-toolchain debs (8 denied fetches) |

### Blocked-egress-tainted failures (agent hit blocks; decisiveness uncertain)

The agent encountered blocked apt/hosts mid-task but may have failed anyway:
code-from-image (3 blocked apt attempts), nginx-request-logging (5),
pytorch-model-cli (1), train-fasttext (4), protein-assembly (4 × 403
host_not_allowed). These are counted as failures in the headline number and
listed here for honesty; deciding whether the block was decisive would require
re-running outside the sandbox.

### Other infra-suspect

- filter-js-from-html: `VerifierTimeoutError` after 900s (verifier's own
  dependency resolution is slow through the proxy); the agent phase completed.
- headless-terminal: fails one interactive-command assertion; a previous
  session attributed this task's failure to sandbox networking, but this run's
  evidence is ambiguous — left as an agent failure.

## Agent failures (~44 tasks)

Everything else: the environment worked, the verifier ran, and the agent's
solution was wrong or incomplete. Notable subcategories:

- **Timeouts/budget exhaustion**: model-extraction-relu-logits
  (`AgentTimeoutError` at 900s); several others hit the harness's soft wall
  clock (840s) or step budget and submitted partial work (e.g.
  chess-best-move, circuit-fibsqrt, code-from-image, caffe-cifar-10).
- **Near-misses** (single verifier test failing): adaptive-rejection-sampler
  (8/9), build-cython-ext (10/11 — the task's own unpinned deps, known from
  the oracle run), cancel-async-tasks (5/6), nginx-request-logging (7/8),
  financial-document-processor (6/7), headless-terminal (6/7).
- **Flaky-by-timing**: largest-eigenval passes here, but in the haiku smoke
  tests it failed `test_speedup[2]` on one of three attempts — a wall-clock
  performance assertion.

## Reproduction notes

Sandbox-specific flags used on every `harbor run`:
`--registry-path .cache/registry.json` (Harbor's registry lookup is blocked)
and `--extra-docker-compose .cache/tls-overlay-astral.yaml` (CA bundle +
astral.sh stand-in + static curl). See DECISIONS.md → "What was verified
where" for the full sandbox story. The raw run directories (`runs/`) are not
committed; `metrics.json`/`matrix.csv` here are the canonical outputs of
`scripts/wolfbench_metrics.py` over the four chunk jobs.
