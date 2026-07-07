#!/usr/bin/env python3
"""Collect compact per-trial rows from Harbor job dirs into one JSON file.

The sandbox container is ephemeral and full run dirs are gitignored, so
after every chunk this extracts what the reports need (reward, tokens,
cache usage, timing) into a small committed artifact. Works for both the
activegraph agent (which writes agent/summary.json) and terminus-2 (token
totals come from Harbor's own result.json agent_result).

Usage:
    python scripts/collect_chunk.py runs/job-a runs/job-b --out results/pass2/a-chunk1.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract_reward(result: dict) -> float | None:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    if "reward" in rewards:
        return float(rewards["reward"])
    if rewards:
        return sum(float(v) for v in rewards.values()) / len(rewards)
    return None


def trial_row(result_path: Path) -> dict:
    result = json.loads(result_path.read_text())
    agent_result = result.get("agent_result") or {}
    row = {
        "task": result["task_name"],
        "trial": result["trial_name"],
        "started_at": result.get("started_at"),
        "reward": extract_reward(result),
        "exception": (result.get("exception_info") or {}).get("exception_type"),
        "n_input_tokens": agent_result.get("n_input_tokens"),
        "n_cache_tokens": agent_result.get("n_cache_tokens"),
        "n_output_tokens": agent_result.get("n_output_tokens"),
    }
    summary_path = result_path.parent / "agent" / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        row.update(
            {
                "finish_reason": summary.get("finish_reason"),
                "steps": summary.get("steps"),
                "tokens_in": summary.get("tokens_in"),
                "tokens_in_uncached": summary.get("tokens_in_uncached"),
                "tokens_cache_read": summary.get("tokens_cache_read"),
                "tokens_cache_creation": summary.get("tokens_cache_creation"),
                "cache_hit_rate": summary.get("cache_hit_rate"),
                "context_version": summary.get("context_version"),
                "tokens_out": summary.get("tokens_out"),
                "wall_time_sec": summary.get("wall_time_sec"),
            }
        )
    return row


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("job_dirs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--label", default=None, help="config label stored with the rows (e.g. A-v1-cache-k3)"
    )
    args = parser.parse_args(argv)

    rows = []
    for job_dir in args.job_dirs:
        if not job_dir.is_dir():
            raise SystemExit(f"not a directory: {job_dir}")
        result_paths = sorted(job_dir.glob("*/result.json"))
        if not result_paths:
            raise SystemExit(f"{job_dir}: no */result.json found")
        rows.extend(trial_row(p) for p in result_paths)

    document = {
        "label": args.label,
        "job_dirs": [str(d) for d in args.job_dirs],
        "n_trials": len(rows),
        "trials": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(document, indent=2) + "\n")
    passed = sum(1 for r in rows if r["reward"] == 1.0)
    print(f"wrote {args.out}: {len(rows)} trials, {passed} passed")


if __name__ == "__main__":
    main(sys.argv[1:])
