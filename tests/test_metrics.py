"""wolfbench_metrics on a synthetic mixed job directory: verifies the five
numbers, the labels, and the missing-reward-counts-as-failure path."""

import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).parent.parent / "scripts" / "wolfbench_metrics.py"
spec = importlib.util.spec_from_file_location("wolfbench_metrics", SCRIPT)
wolfbench_metrics = importlib.util.module_from_spec(spec)
sys.modules["wolfbench_metrics"] = wolfbench_metrics
spec.loader.exec_module(wolfbench_metrics)


def write_trial(job_dir: Path, task: str, trial: str, started_at: str, reward):
    trial_dir = job_dir / trial
    trial_dir.mkdir(parents=True)
    result = {
        "task_name": task,
        "trial_name": trial,
        "started_at": started_at,
    }
    if reward is not None:
        result["verifier_result"] = {"rewards": {"reward": reward}}
    else:
        result["exception_info"] = {"exception_type": "AgentTimeoutError"}
    (trial_dir / "result.json").write_text(json.dumps(result))


def test_mixed_matrix(tmp_path):
    job = tmp_path / "job"
    # alpha: passes both runs -> always
    write_trial(job, "alpha", "alpha__a1", "2026-01-01T00:00:00", 1.0)
    write_trial(job, "alpha", "alpha__a2", "2026-01-01T01:00:00", 1.0)
    # beta: passes only the second run -> sometimes
    write_trial(job, "beta", "beta__b1", "2026-01-01T00:00:00", 0.0)
    write_trial(job, "beta", "beta__b2", "2026-01-01T01:00:00", 1.0)
    # gamma: one fail, one missing reward (exception) -> never
    write_trial(job, "gamma", "gamma__g1", "2026-01-01T00:00:00", 0.0)
    write_trial(job, "gamma", "gamma__g2", "2026-01-01T01:00:00", None)

    trials, problems = wolfbench_metrics.load_trials([job])
    assert len(trials) == 6
    assert len(problems) == 1 and "gamma__g2" in problems[0]

    tasks, n_runs, matrix = wolfbench_metrics.build_matrix(trials)
    assert tasks == ["alpha", "beta", "gamma"]
    assert n_runs == 2
    assert matrix["alpha"] == [True, True]
    assert matrix["beta"] == [False, True]
    assert matrix["gamma"] == [False, False]

    metrics = wolfbench_metrics.compute_metrics(tasks, n_runs, matrix)
    assert metrics["solid"] == round(1 / 3, 4)
    assert metrics["ceiling"] == round(2 / 3, 4)
    assert metrics["average"] == 0.5
    assert metrics["worst_of"] == round(1 / 3, 4)  # run 1: only alpha passes
    assert metrics["best_of"] == round(2 / 3, 4)  # run 2: alpha and beta pass
    assert metrics["spread"] == round(2 / 3 - 1 / 3, 4)

    out = tmp_path / "out"
    wolfbench_metrics.write_outputs(out, metrics, tasks, n_runs, matrix, problems)
    lines = (out / "matrix.csv").read_text().strip().splitlines()
    assert lines[0] == "task,run_1,run_2,label"
    assert lines[1] == "alpha,1,1,always"
    assert lines[2] == "beta,0,1,sometimes"
    assert lines[3] == "gamma,0,0,never"
    saved = json.loads((out / "metrics.json").read_text())
    assert saved["incomplete_trials"] == problems
