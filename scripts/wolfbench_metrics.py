#!/usr/bin/env python3
"""WolfBench metrics over Harbor run directories.

Input: one or more Harbor job directories (each containing per-trial
subdirectories with a result.json, as written by `harbor run`). All trials
for the same task, across all given directories, are ordered by start time
and become that task's run columns.

Output (written to --out-dir, printed to stdout as well):
- metrics.json: the five WolfBench numbers plus counts and the spread.
- matrix.csv: tasks as rows, runs as columns, 0/1 cells, plus a per-task
  label (always / sometimes / never).

Definitions, with pass = verifier reward of 1.0 for the trial:
- solid:    fraction of tasks that pass in ALL runs (the floor you can trust)
- worst_of: the per-run pass rate of the worst single run
- average:  mean pass rate over all cells
- best_of:  the per-run pass rate of the best single run
- ceiling:  fraction of tasks that pass in ANY run (what is possible)
- spread:   ceiling minus solid (the variance meter)

Trials whose result.json is missing a verifier reward (exception, timeout
before verification) count as failures; they are also listed under
"incomplete_trials" in metrics.json so nothing fails silently.

Usage:
    python scripts/wolfbench_metrics.py runs/my-job [runs/other-job ...] \
        --out-dir runs/my-job
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PASS_THRESHOLD = 1.0 - 1e-9


def load_trials(job_dirs: list[Path]) -> tuple[list[dict], list[str]]:
    """Collect (task_name, trial_name, started_at, passed) for every trial.

    Returns (trials, problems). Problems are human-readable notes about
    trials that could not be scored; they count as failures.
    """
    trials: list[dict] = []
    problems: list[str] = []
    for job_dir in job_dirs:
        result_paths = sorted(job_dir.glob("*/result.json"))
        if not result_paths:
            problems.append(f"{job_dir}: no */result.json found")
            continue
        for path in result_paths:
            try:
                result = json.loads(path.read_text())
            except (OSError, json.JSONDecodeError) as exc:
                problems.append(f"{path}: unreadable ({exc}); counted as failure")
                trials.append(
                    {
                        "task": path.parent.name.split("__")[0],
                        "trial": path.parent.name,
                        "started_at": "",
                        "passed": False,
                    }
                )
                continue
            reward = extract_reward(result)
            if reward is None:
                exception = (result.get("exception_info") or {}).get(
                    "exception_type", "no reward"
                )
                problems.append(
                    f"{path.parent.name}: no verifier reward ({exception}); "
                    "counted as failure"
                )
            trials.append(
                {
                    "task": result["task_name"],
                    "trial": result["trial_name"],
                    "started_at": result.get("started_at") or "",
                    "passed": reward is not None and reward >= PASS_THRESHOLD,
                }
            )
    return trials, problems


def extract_reward(result: dict) -> float | None:
    verifier_result = result.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    if "reward" in rewards:
        return float(rewards["reward"])
    if rewards:
        # No canonical "reward" key: fall back to the mean of all rewards.
        return sum(float(v) for v in rewards.values()) / len(rewards)
    return None


def build_matrix(trials: list[dict]) -> tuple[list[str], int, dict[str, list[bool]]]:
    """Group trials into a task-by-run boolean matrix.

    Runs are ordered per task by (started_at, trial_name). Tasks with
    fewer trials than the widest task get explicit False cells and a
    problem note upstream would have flagged the cause.
    """
    by_task: dict[str, list[dict]] = {}
    for trial in trials:
        by_task.setdefault(trial["task"], []).append(trial)
    tasks = sorted(by_task)
    n_runs = max((len(v) for v in by_task.values()), default=0)
    matrix: dict[str, list[bool]] = {}
    for task in tasks:
        ordered = sorted(by_task[task], key=lambda t: (t["started_at"], t["trial"]))
        row = [t["passed"] for t in ordered]
        row += [False] * (n_runs - len(row))
        matrix[task] = row
    return tasks, n_runs, matrix


def compute_metrics(
    tasks: list[str], n_runs: int, matrix: dict[str, list[bool]]
) -> dict:
    n_tasks = len(tasks)
    if n_tasks == 0 or n_runs == 0:
        raise SystemExit("no trials found; nothing to compute")
    per_run_rates = [
        sum(matrix[task][run] for task in tasks) / n_tasks for run in range(n_runs)
    ]
    solid = sum(all(matrix[task]) for task in tasks) / n_tasks
    ceiling = sum(any(matrix[task]) for task in tasks) / n_tasks
    average = sum(sum(matrix[task]) for task in tasks) / (n_tasks * n_runs)
    return {
        "solid": round(solid, 4),
        "worst_of": round(min(per_run_rates), 4),
        "average": round(average, 4),
        "best_of": round(max(per_run_rates), 4),
        "ceiling": round(ceiling, 4),
        "spread": round(ceiling - solid, 4),
        "n_tasks": n_tasks,
        "n_runs": n_runs,
        "per_run_pass_rates": [round(r, 4) for r in per_run_rates],
    }


def task_label(row: list[bool]) -> str:
    if all(row):
        return "always"
    if any(row):
        return "sometimes"
    return "never"


def write_outputs(
    out_dir: Path,
    metrics: dict,
    tasks: list[str],
    n_runs: int,
    matrix: dict[str, list[bool]],
    problems: list[str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({**metrics, "incomplete_trials": problems}, indent=2) + "\n"
    )

    matrix_path = out_dir / "matrix.csv"
    with matrix_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + [f"run_{i + 1}" for i in range(n_runs)] + ["label"])
        for task in tasks:
            row = matrix[task]
            writer.writerow([task] + [int(cell) for cell in row] + [task_label(row)])

    print(f"wrote {metrics_path}")
    print(f"wrote {matrix_path}")
    print(json.dumps(metrics, indent=2))
    if problems:
        print(f"\n{len(problems)} trial(s) counted as failures without a reward:")
        for problem in problems:
            print(f"  - {problem}")


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "job_dirs", nargs="+", type=Path, help="Harbor job directories"
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("."),
        help="Where to write metrics.json and matrix.csv (default: cwd)",
    )
    args = parser.parse_args(argv)

    for job_dir in args.job_dirs:
        if not job_dir.is_dir():
            raise SystemExit(f"not a directory: {job_dir}")

    trials, problems = load_trials(args.job_dirs)
    tasks, n_runs, matrix = build_matrix(trials)
    metrics = compute_metrics(tasks, n_runs, matrix)
    write_outputs(args.out_dir, metrics, tasks, n_runs, matrix, problems)


if __name__ == "__main__":
    main(sys.argv[1:])
