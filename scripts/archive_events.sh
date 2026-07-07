#!/usr/bin/env bash
# Archive the durable per-trial evidence for a harbor job into one
# compressed tarball under results/events/, sized for git.
#
# Persistence policy (added 2026-07-06 after pass-2 raw transcripts were
# lost to container reclaims): summaries alone are not enough. The event
# log IS the run; every job's event stores get committed alongside the
# compact chunk artifact, immediately, in the same push.
#
# Includes per trial: event_log.json (full ordered events including the
# context_built audit trail), summary.json, config.json, trial.log.
# Excludes: raw docker artifacts and verifier scratch (regenerable or
# environment-owned, not agent evidence).
#
# Usage: bash scripts/archive_events.sh runs/<job-name>
set -euo pipefail
cd "$(dirname "$0")/.."

job_dir="${1:?usage: archive_events.sh runs/<job-name>}"
job_name="$(basename "$job_dir")"
out_dir=results/events
out="${out_dir}/${job_name}-events.tar.gz"
mkdir -p "$out_dir"

files=$(cd "$job_dir" && find . -maxdepth 3 \( \
    -name event_log.json -o -name summary.json -o -name config.json \
    -o -name trajectory.json -o -name trial.log \) | sort)
if [ -z "$files" ]; then
    echo "FATAL: no evidence files under $job_dir" >&2
    exit 1
fi
# shellcheck disable=SC2086
tar czf "$out" -C "$job_dir" $files
echo "archived $(echo "$files" | wc -l) files -> $out ($(du -h "$out" | cut -f1))"
