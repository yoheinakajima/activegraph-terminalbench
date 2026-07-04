#!/usr/bin/env bash
# Run one pass-2 config over task-chunk files, sequentially and resumably.
#
# Per chunk: refuse if another harbor run is alive, clean docker (safe:
# nothing running), check disk headroom, run harbor, collect the compact
# artifact, commit and push it. A chunk whose collected artifact already
# exists is skipped, so rerunning after a container reclaim resumes where
# the commits left off.
#
# Usage:
#   AGENT_FLAGS="--agent ... --model anthropic/claude-sonnet-4-5 [--ak k=v]" \
#   [K=3] scripts/run_config.sh LABEL chunkfile...
set -euo pipefail

LABEL="${1:?usage: run_config.sh LABEL chunkfile...}"
shift
: "${AGENT_FLAGS:?set AGENT_FLAGS (agent + model flags for harbor run)}"
K="${K:-3}"
MIN_FREE_GB="${MIN_FREE_GB:-16}"
OVERLAY="${OVERLAY:-.cache/tls-overlay-astral.yaml}"

for chunk in "$@"; do
    name="pass2-${LABEL}-$(basename "${chunk%.txt}")"
    out="results/pass2/${name}.json"
    if [ -f "$out" ]; then
        echo "skip ${name}: ${out} already collected"
        continue
    fi
    # exclusive runner lock: held for the life of this script; prevents a
    # second runner from cleaning docker under a live harbor run. Immune to
    # command-line text (pgrep self-matched orchestrating shells).
    if [ -z "${PASS2_LOCK_HELD:-}" ]; then
        exec 9> /tmp/pass2-runner.lock
        flock -n 9 || { echo "FATAL: another runner holds the lock" >&2; exit 1; }
        export PASS2_LOCK_HELD=1
    fi
    docker ps -aq | xargs -r docker rm -f > /dev/null 2>&1 || true
    docker image prune -af > /dev/null
    free_gb=$(( $(df --output=avail / | tail -1) / 1048576 ))
    if [ "$free_gb" -lt "$MIN_FREE_GB" ]; then
        echo "FATAL: only ${free_gb}GB free (< ${MIN_FREE_GB}GB) before ${name}" >&2
        exit 1
    fi
    task_args=$(sed 's/^/-i /' "$chunk" | tr '\n' ' ')
    echo "=== ${name}: $(wc -l < "$chunk") tasks, k=${K}, ${free_gb}GB free ==="
    # shellcheck disable=SC2086
    uv run harbor run --dataset terminal-bench@2.0 ${AGENT_FLAGS} ${task_args} \
        -k "$K" --n-concurrent 4 --jobs-dir runs --job-name "$name" \
        --registry-path .cache/registry.json \
        --extra-docker-compose "$OVERLAY"
    uv run python scripts/collect_chunk.py "runs/$name" --out "$out" --label "$LABEL"
    git add "$out"
    git commit -q -m "pass2: collect ${name}"
    git push -q origin HEAD || git push origin HEAD
    # Keep full trial evidence (event logs, verifier output) compressed on
    # disk for the per-task analysis; ~10x smaller than the live dir.
    mkdir -p runs/archives
    tar -czf "runs/archives/${name}.tar.gz" -C runs "$name"
    rm -rf "runs/$name"
done
echo "config ${LABEL}: all requested chunks done"
