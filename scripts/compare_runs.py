#!/usr/bin/env python3
"""Compare pass 2 run configs from collected chunk artifacts.

Consumes the JSON files written by collect_chunk.py (the committed,
reclaim-proof form of the runs), not raw Harbor job dirs. Emits:

- a metrics JSON: per config, the five WolfBench numbers (solid, worst_of,
  average, best_of, ceiling), token totals, cache hit rate, and dollar
  cost, computed on both the full 89-task basis and the clean subset
  derived from results/excluded_tasks.json;
- per-config matrix CSVs (task x run cells);
- pairwise per-task diff CSVs (e.g. C vs B): who solves what.

Usage:
    python scripts/compare_runs.py \
        --config A=results/pass2/A-c1.json,results/pass2/A-c2.json \
        --config B=results/pass2/B-c1.json \
        --diff C:B --diff A:B \
        --out-dir reports/pass2_artifacts \
        --metrics-out reports/pass2_metrics.json

Pricing: sonnet 4.5 dollars per MTok. For the activegraph agent the cache
split is exact (uncached/write/read from summary.json). terminus-2 rows
only report total input and cached-read tokens (LiteLLM does not surface
cache-write counts), so its cost treats non-read input as plain input and
skips the 1.25x write premium: a slight underestimate, called out in the
output as cost_is_lower_bound.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

PRICE_IN = 3.0
PRICE_OUT = 15.0
PRICE_CACHE_READ = 0.3
PRICE_CACHE_WRITE = 3.75

EXCLUDED_TASKS_PATH = Path("results/excluded_tasks.json")
PASS_THRESHOLD = 1.0 - 1e-9


def load_config_trials(paths: list[Path]) -> list[dict]:
    trials: list[dict] = []
    for path in paths:
        document = json.loads(path.read_text())
        trials.extend(document["trials"])
    return trials


def build_matrix(trials: list[dict]) -> tuple[dict[str, list[bool]], int, list[str]]:
    """task -> ordered pass/fail cells, padded to the widest run count."""
    problems: list[str] = []
    by_task: dict[str, list[dict]] = {}
    for trial in trials:
        by_task.setdefault(trial["task"], []).append(trial)
    n_runs = max((len(v) for v in by_task.values()), default=0)
    matrix: dict[str, list[bool]] = {}
    for task, rows in sorted(by_task.items()):
        rows.sort(key=lambda r: (r.get("started_at") or "", r["trial"]))
        cells = [
            r["reward"] is not None and r["reward"] >= PASS_THRESHOLD for r in rows
        ]
        for r in rows:
            if r["reward"] is None:
                problems.append(
                    f"{r['trial']}: no reward ({r.get('exception')}); counted as failure"
                )
        if len(cells) < n_runs:
            problems.append(f"{task}: only {len(cells)}/{n_runs} trials; padded as failure")
            cells += [False] * (n_runs - len(cells))
        matrix[task] = cells
    return matrix, n_runs, problems


def five_metrics(matrix: dict[str, list[bool]], n_runs: int) -> dict:
    tasks = sorted(matrix)
    n = len(tasks)
    if n == 0 or n_runs == 0:
        raise SystemExit("empty matrix")
    per_run = [sum(matrix[t][r] for t in tasks) / n for r in range(n_runs)]
    return {
        "n_tasks": n,
        "n_runs": n_runs,
        "solid": round(sum(all(matrix[t]) for t in tasks) / n, 4),
        "worst_of": round(min(per_run), 4),
        "average": round(sum(sum(matrix[t]) for t in tasks) / (n * n_runs), 4),
        "best_of": round(max(per_run), 4),
        "ceiling": round(sum(any(matrix[t]) for t in tasks) / n, 4),
        "per_run_pass_rates": [round(r, 4) for r in per_run],
    }


def cost_and_tokens(trials: list[dict]) -> dict:
    has_split = [t for t in trials if t.get("tokens_in_uncached") is not None]
    plain = [t for t in trials if t.get("tokens_in_uncached") is None]
    uncached = sum(t["tokens_in_uncached"] for t in has_split)
    writes = sum(t["tokens_cache_creation"] for t in has_split)
    reads = sum(t["tokens_cache_read"] for t in has_split)
    out_split = sum(t.get("tokens_out") or 0 for t in has_split)
    cost = (
        uncached * PRICE_IN + writes * PRICE_CACHE_WRITE + reads * PRICE_CACHE_READ
    ) / 1e6 + out_split * PRICE_OUT / 1e6

    plain_in = sum(t.get("n_input_tokens") or 0 for t in plain)
    plain_reads = sum(t.get("n_cache_tokens") or 0 for t in plain)
    plain_out = sum(t.get("n_output_tokens") or 0 for t in plain)
    cost += (
        (plain_in - plain_reads) * PRICE_IN + plain_reads * PRICE_CACHE_READ
    ) / 1e6 + plain_out * PRICE_OUT / 1e6

    total_in = sum(t.get("tokens_in") or t.get("n_input_tokens") or 0 for t in trials)
    total_reads = reads + plain_reads
    total_out = out_split + plain_out
    n_solved = sum(
        1 for t in trials if t["reward"] is not None and t["reward"] >= PASS_THRESHOLD
    )
    return {
        "tokens_in": total_in,
        "tokens_out": total_out,
        "tokens_cache_read": total_reads,
        "cache_hit_rate": round(total_reads / total_in, 4) if total_in else 0.0,
        "cost_usd": round(cost, 2),
        "cost_is_lower_bound": bool(plain),
        "n_trials": len(trials),
        "n_solved_trials": n_solved,
        "tokens_per_solve": round(total_in / n_solved) if n_solved else None,
    }


def restrict(matrix: dict[str, list[bool]], tasks: set[str]) -> dict[str, list[bool]]:
    return {t: v for t, v in matrix.items() if t in tasks}


def write_matrix_csv(path: Path, matrix: dict[str, list[bool]], n_runs: int) -> None:
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task"] + [f"run_{i + 1}" for i in range(n_runs)] + ["label"])
        for task, cells in sorted(matrix.items()):
            label = "always" if all(cells) else ("sometimes" if any(cells) else "never")
            writer.writerow([task] + [int(c) for c in cells] + [label])


def write_diff_csv(
    path: Path,
    left: str,
    right: str,
    left_matrix: dict[str, list[bool]],
    right_matrix: dict[str, list[bool]],
) -> None:
    tasks = sorted(set(left_matrix) | set(right_matrix))
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "task",
                f"{left}_passes",
                f"{left}_trials",
                f"{right}_passes",
                f"{right}_trials",
                "category",
            ]
        )
        for task in tasks:
            lcells = left_matrix.get(task, [])
            rcells = right_matrix.get(task, [])
            lp, rp = sum(lcells), sum(rcells)
            lflip = 0 < lp < len(lcells)
            rflip = 0 < rp < len(rcells)
            if lp and not rp:
                category = f"{left}_only"
            elif rp and not lp:
                category = f"{right}_only"
            elif lp and rp:
                category = "both"
            else:
                category = "neither"
            if lflip or rflip:
                flips = "+".join(
                    name for name, flip in ((left, lflip), (right, rflip)) if flip
                )
                category += f" (flaky:{flips})"
            writer.writerow([task, lp, len(lcells), rp, len(rcells), category])


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--config",
        action="append",
        required=True,
        metavar="LABEL=file1,file2",
        help="collected chunk JSONs for one config",
    )
    parser.add_argument("--diff", action="append", default=[], metavar="LEFT:RIGHT")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--excluded", type=Path, default=EXCLUDED_TASKS_PATH)
    args = parser.parse_args(argv)

    excluded_doc = json.loads(args.excluded.read_text())
    excluded = {e["task"] for e in excluded_doc["excluded"]}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"configs": {}, "excluded_tasks": sorted(excluded)}
    matrices: dict[str, dict[str, list[bool]]] = {}

    for spec in args.config:
        label, _, files = spec.partition("=")
        if not files:
            raise SystemExit(f"bad --config spec: {spec!r}")
        trials = load_config_trials([Path(f) for f in files.split(",")])
        matrix, n_runs, problems = build_matrix(trials)
        matrices[label] = matrix
        clean = restrict(matrix, set(matrix) - excluded)
        report["configs"][label] = {
            "full_basis": five_metrics(matrix, n_runs),
            "clean_basis": five_metrics(clean, n_runs) if clean else None,
            **cost_and_tokens(trials),
            "problems": problems,
        }
        write_matrix_csv(args.out_dir / f"matrix_{label}.csv", matrix, n_runs)
        print(f"{label}: {len(matrix)} tasks x {n_runs} runs, "
              f"avg {report['configs'][label]['full_basis']['average']}")

    for spec in args.diff:
        left, _, right = spec.partition(":")
        if left not in matrices or right not in matrices:
            raise SystemExit(f"--diff {spec!r} names an unknown config")
        path = args.out_dir / f"diff_{left}_vs_{right}.csv"
        write_diff_csv(path, left, right, matrices[left], matrices[right])
        print(f"wrote {path}")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {args.metrics_out}")


if __name__ == "__main__":
    main(sys.argv[1:])
