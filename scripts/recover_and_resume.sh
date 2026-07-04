#!/usr/bin/env bash
# One-command recovery after a container reset: restore the branch from
# origin, rebuild the ephemeral sandbox, and resume the phase-2 queue.
# run_config.sh skips chunks whose artifacts are already collected, so
# this is idempotent and loses at most the in-flight chunk.
#
# Usage: bash scripts/recover_and_resume.sh
set -euo pipefail
cd "$(dirname "$0")/.."

BRANCH=claude/activegraph-harness-benchmark-eiyeek
git fetch origin "$BRANCH"
git reset --hard "origin/$BRANCH"
bash scripts/rebuild_sandbox.sh

if pgrep -f "harbor run" > /dev/null; then
    echo "harbor already running; not relaunching"
    exit 0
fi

CH=results/pass2/chunks
FINE=""
for c in s06a s06b s07a s07b s08a s08b s09a s09b s10a s10b s11a s11b g1 g2 g3 g4; do
    FINE="$FINE $CH/$c.txt"
done

# shellcheck disable=SC2086
nohup bash -c "
export ANTHROPIC_API_KEY=\"\$ANTHROP_API_KEY\"
AGENT_FLAGS='--agent activegraph_harness.agent:ActiveGraphAgent --model anthropic/claude-sonnet-4-5 --ak enable_cache=true' bash scripts/run_config.sh A $FINE
AGENT_FLAGS='--agent activegraph_harness.agent:ActiveGraphAgent --model anthropic/claude-sonnet-4-5 --ak context_version=v2' bash scripts/run_config.sh C $CH/s04.txt $CH/s05.txt $FINE
" > /tmp/run-phase2.log 2>&1 &
echo "phase-2 queue resumed"
