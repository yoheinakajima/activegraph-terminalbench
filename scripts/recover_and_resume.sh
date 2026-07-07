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

PIDFILE=/tmp/pass2-queue.pid
if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2> /dev/null; then
    echo "queue already running (pid $(cat "$PIDFILE")); not relaunching"
    exit 0
fi

CH=results/pass2/chunks
FINE=""
# g1-g3 (hf-model-inference, mteb-leaderboard, mteb-retrieve) dropped:
# all three are in results/excluded_tasks.json (huggingface.co blocked by
# sandbox egress), guaranteed zeros outside the clean subset, and g1's
# docker build cannot even complete behind the proxy (pip TLS failure
# repeatedly crashed the queue).
for c in s06a s06b s07a s07b s08a s08b s09a s09b s10a s10b s11a s11b g4; do
    FINE="$FINE $CH/$c.txt"
done

# shellcheck disable=SC2086
nohup bash -c "
export ANTHROPIC_API_KEY=\"\$ANTHROP_API_KEY\"
AGENT_FLAGS='--agent activegraph_harness.agent:ActiveGraphAgent --model anthropic/claude-sonnet-4-5 --ak enable_cache=true' bash scripts/run_config.sh A $FINE
AGENT_FLAGS='--agent activegraph_harness.agent:ActiveGraphAgent --model anthropic/claude-sonnet-4-5 --ak context_version=v2' bash scripts/run_config.sh C $CH/s04.txt $CH/s05.txt $FINE
" > /tmp/run-phase2.log 2>&1 &
echo $! > "$PIDFILE"
echo "phase-2 queue resumed"
