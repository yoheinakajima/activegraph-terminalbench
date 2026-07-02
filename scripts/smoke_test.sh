#!/usr/bin/env bash
# Smoke test: (1) oracle run to prove Docker + Harbor work, no LLM, no cost;
# (2) a 2-task run of the ActiveGraph agent with claude-haiku-4-5 (cheap).
# Run from the repo root. Requires Docker running; step 2 requires
# ANTHROPIC_API_KEY.
set -euo pipefail

echo "== Step 1: oracle smoke test (no LLM, no cost) =="
uv run harbor run \
  --dataset terminal-bench@2.0 \
  --agent oracle \
  -i largest-eigenval -i fix-code-vulnerability \
  --n-concurrent 2 \
  --jobs-dir runs \
  --job-name oracle-smoke

echo "== Step 2: agent smoke test on 2 easy tasks with haiku =="
if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "ANTHROPIC_API_KEY is not set; skipping the agent run." >&2
  exit 1
fi
uv run harbor run \
  --dataset terminal-bench@2.0 \
  --agent activegraph_harness.agent:ActiveGraphAgent \
  --model anthropic/claude-haiku-4-5 \
  -i largest-eigenval -i fix-code-vulnerability \
  --n-concurrent 2 \
  --jobs-dir runs \
  --job-name agent-smoke

echo "== Event logs =="
ls runs/agent-smoke/*/agent/

echo "== Metrics =="
uv run python scripts/wolfbench_metrics.py runs/agent-smoke --out-dir runs/agent-smoke
