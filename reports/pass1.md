# Pass 1: activegraph-harness on Terminal-Bench 2.0

Written 2026-07-03, after pass 1 completed and before any pass 2 code
changes. Every number here comes from committed artifacts: `results/`
(metrics.json, matrix.csv, README.md), `DECISIONS.md`, and the git history
of this branch. The raw Harbor run directories lived in the (gitignored,
since reclaimed) sandbox container; their aggregates were committed.

## Setup

The harness is a minimal ReAct agent for Harbor / Terminal-Bench 2.0 built
on an ActiveGraph event store: every model turn, command, observation, and
error in a trial is a typed, linked event, and everything the model sees is
assembled by a single retrieval seam, `build_context()`. The v1
build_context is deliberately dumb: system prompt, task instruction, and a
verbatim tail of the most recent transcript exchanges (at most 20), all
reprojected from the event log each turn. Pass 1 ran the full 89-task
Terminal-Bench 2.0 dataset once (k=1) with `anthropic/claude-sonnet-4-6`,
4 trials concurrent, inside a Claude Code cloud sandbox on 2026-07-03. The
sandbox matters: all egress goes through an allowlisting proxy (PyPI, npm,
crates.io, and GitHub-from-containers work; Debian/Ubuntu apt mirrors,
huggingface.co, astral.sh, and most other hosts are blocked), TLS is
intercepted, and disk was about 30 GB of headroom against task images up to
15 GB. Local workarounds (a stand-in astral.sh serving the real uv 0.9.5
binaries repacked from PyPI wheels, a static curl mounted into every
container, a combined CA bundle) made 82 of 89 verifiers runnable at all.

## Headline numbers

Three framings of the same run, from `results/metrics.json` and
`results/README.md`:

| framing | number | what it treats as a failure |
|---|---|---|
| canonical | 34/89 = 38.2% | everything, including trials whose Docker environment never started |
| infra-recovered | 36/89 = 40.4% | as canonical, but the five env-startup failures were retried once on clean disk and the two that then passed are counted |
| clean subset | 36/82 = 43.9% | as infra-recovered, excluding the 7 tasks that provably cannot be attempted behind this sandbox's egress policy |

Which framing to use: the canonical number is the honest answer to "what
did this exact run score"; the infra-recovered number is the honest answer
to "what did the agent score when it actually got to run" (the retries were
first attempts, not second chances, because the environment had crashed
before the agent ever started); the clean-subset number is the only one
that should be compared, directionally, against results from unrestricted
infrastructure, and even it still contains five more egress-tainted tasks
(see taxonomy) that could not be proven either way.

## Cost and token profile

Total API spend was about $112: $109.17 for the 89 sonnet trials
(29.5M input tokens, 1.39M output tokens, estimated at $3/$15 per MTok)
plus $2.36 of haiku smoke tests. Roughly 80% of the sonnet spend was input
tokens, and none of it was cached. The cause is structural: the v1 ReAct
loop resends the entire growing transcript on every turn, so a 60-turn
trial pays for its history about 60 times over. Median trials were fine;
long trials (the loop allows up to 4096-token responses and a 900 s task
budget) reached 900k to 1M input tokens each. This is the single biggest
lever pass 2 pulls.

## External context

Published Terminal-Bench 2.0 leaderboard results on Claude Sonnet 4.5, for
reference. These are k=5 runs with error bars, on a different model version
(4.5, not our 4.6) and unrestricted infrastructure, so the comparison is
directional only:

| agent | pass rate (Sonnet 4.5, k=5) |
|---|---|
| Warp (multiple models) | 50.1 ± 2.7 |
| CAMEL-AI | 46.5 ± 2.4 |
| Goose | 43.1 ± 2.6 |
| Terminus 2 | 42.8 ± 2.8 |
| OpenHands | 42.6 ± 2.8 |
| Mini-SWE-Agent | 42.5 ± 2.8 |
| Claude Code | 40.1 ± 2.9 |

Our clean-subset 43.9% sits in the middle of that board, but on one flip of
a k=1 coin, on a newer model, minus 7 tasks. It is a plausibility check,
not a ranking claim.

## Failure taxonomy

From `results/README.md`, which itemizes every failed task with evidence.

Sandbox-infra, provable (7): hf-model-inference, mteb-retrieve,
mteb-leaderboard, and count-dataset-tokens all show denied huggingface.co
requests in the agent transcript (`host_not_allowed` from the egress
proxy); reshard-c4-data's own verifier needs the allenai/c4 dataset from
huggingface; build-pmars's verifier asserts Debian source packages were
used, which is impossible with apt mirrors blocked; make-doom-for-mips was
blocked downloading the gcc-12 MIPS cross-toolchain (8 denied fetches).

Env-startup failures (5): custom-memory-heap-crash, pytorch-model-recovery,
hf-model-inference, mteb-retrieve, mteb-leaderboard errored before the
agent ran, from Docker image-extract races (a disk-space guard pruning
concurrently with pulls) and then ENOSPC on ~15 GB torch/mteb images. All
five were retried once, alone, on clean disk. Result: two passed outright
(custom-memory-heap-crash, pytorch-model-recovery, both 1.0), and three ran
into the huggingface wall above. The retry exercise is what separated
"sandbox broke the environment" from "sandbox blocks the task itself".

Egress-tainted, ambiguous (5): code-from-image, nginx-request-logging,
pytorch-model-cli, train-fasttext, protein-assembly. The agent hit blocked
apt or 403 responses mid-task, but whether the block was decisive is
unproven (nginx-request-logging still passed 7 of 8 verifier tests). These
are counted as plain failures in every headline number.

Agent failures (~44): environment fine, verifier ran, solution wrong or
incomplete. Two dominant modes. First, plain wrong or partial solutions.
Second, budget exhaustion: several trials hit the harness's 840 s soft
wall clock or the step budget and submitted partial work
(chess-best-move, circuit-fibsqrt, caffe-cifar-10, build-pov-ray among
them), and model-extraction-relu-logits hit Harbor's hard 900 s
AgentTimeoutError. Notable single-test near-misses: adaptive-rejection-
sampler failed 1 of 9 verifier tests, cancel-async-tasks 1 of 6,
financial-document-processor 1 of 7, build-cython-ext 1 of 11 (that one
test is known to fail from the task's own unpinned dependencies, observed
in the pass 1 oracle run too).

## Limitations

Stated plainly. k=1 is a coin flip by our own five-metric thesis: the whole
point of the solid/worst/average/best/ceiling spread is that single-run
pass rates on agentic benchmarks are noisy, and pass 1 produced exactly one
sample per task (the k=2 haiku smoke test already showed largest-eigenval
flipping 1 to 0 on a wall-clock speedup assertion). The model is Sonnet
4.6 while the published comparison board is Sonnet 4.5. The sandbox's
broken egress depresses the score by an amount we can bound only partially
(7 tasks provably, 5 more plausibly). And there is no matched control run:
no other agent was run on this model on this infrastructure, so pass 1
supports no causal claim about the harness design at all.

## What pass 2 changes and why

Pass 2 adds prompt caching (the transcript-resend cost structure is the
expensive part, and caching makes the v1 baseline cheap enough to run at
k=3), implements the v2 graph-retrieval build_context behind the existing
seam (small retrieved context instead of a big cached transcript, the
design the event store was built for), and runs a matched matrix on Sonnet
4.5: v1 baseline, terminus-2 as a same-model same-infra control, and v2 as
the treatment. The control both anchors us to the published 42.8 ± 2.8 and
prices the remaining infrastructure tax.
