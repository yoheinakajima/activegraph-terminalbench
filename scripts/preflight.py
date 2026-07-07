#!/usr/bin/env python3
"""Preflight for benchmark runs in the disk-constrained sandbox.

Pass 1 lost five trials to environment-startup failures that were
self-inflicted: a background disk guard ran `docker image prune -af` while
pulls were extracting (corrupting them), and two ~15 GB images hit ENOSPC
with other images still resident. The fixes this script encodes:

1. Never prune while anything is running. This script refuses to start if
   a harbor process or a prune is alive, and does its own prune only then.
2. Serialize the known-large-image pulls instead of letting 4 concurrent
   trials pull them together.
3. Check disk headroom before every pull and refuse loudly below the
   threshold.

Usage:
    python scripts/preflight.py --min-free-gb 20 \
        --prepull hf-model-inference mteb-leaderboard ...

--prepull resolves each task name to its task.toml docker_image in the
harbor cache and pulls sequentially with a headroom check between pulls.
With --verify-start it also does `docker compose up --wait` + down on a
minimal compose file per image, proving the environment can start, then
removes the image again to return the disk (use for the one-off check of
the pass 1 casualties, not before real runs).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path

HARBOR_TASK_CACHE = Path.home() / ".cache" / "harbor" / "tasks"


def fail(message: str) -> None:
    raise SystemExit(f"preflight FAILED: {message}")


def free_gb(path: str = "/") -> float:
    return shutil.disk_usage(path).free / 1e9


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, check=False, text=True, capture_output=True, **kwargs)


def assert_nothing_running() -> None:
    probe = run(["pgrep", "-af", "harbor run|docker.*prune"])
    lines = [
        line
        for line in probe.stdout.splitlines()
        if "pgrep" not in line and line.strip()
    ]
    if lines:
        fail(f"refusing to touch docker while these are running:\n{probe.stdout}")


def resolve_image(task_name: str) -> str:
    matches = list(HARBOR_TASK_CACHE.glob(f"*/{task_name}/task.toml"))
    if not matches:
        fail(f"task {task_name!r} not found under {HARBOR_TASK_CACHE}")
    config = tomllib.loads(matches[0].read_text())
    image = config.get("environment", {}).get("docker_image")
    if not image:
        fail(f"{matches[0]} has no [environment] docker_image")
    return image


def pull_serialized(image: str, min_free_gb: float) -> None:
    headroom = free_gb()
    print(f"disk headroom before pull: {headroom:.1f} GB", flush=True)
    if headroom < min_free_gb:
        fail(
            f"only {headroom:.1f} GB free (< {min_free_gb} GB) before pulling "
            f"{image}; prune or grow the disk first"
        )
    result = run(["docker", "pull", image])
    if result.returncode != 0:
        fail(f"docker pull {image} failed:\n{result.stderr[-2000:]}")
    print(f"pulled {image}", flush=True)


def verify_start(image: str) -> None:
    compose = f"""services:
  main:
    image: {image}
    command: ["sh", "-c", "echo started && sleep 2"]
"""
    with tempfile.TemporaryDirectory() as tmp:
        compose_path = Path(tmp) / "docker-compose.yaml"
        compose_path.write_text(compose)
        up = run(
            ["docker", "compose", "-f", str(compose_path), "up", "--wait", "--detach"]
        )
        run(["docker", "compose", "-f", str(compose_path), "down", "-t", "1"])
        if up.returncode != 0:
            fail(f"compose up failed for {image}:\n{up.stderr[-2000:]}")
    print(f"verified start: {image}", flush=True)


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--min-free-gb", type=float, default=20.0)
    parser.add_argument("--prepull", nargs="*", default=[], metavar="TASK")
    parser.add_argument(
        "--verify-start",
        action="store_true",
        help="compose up/down each prepulled image, then remove it again",
    )
    parser.add_argument(
        "--no-prune",
        action="store_true",
        help="skip the initial (safe, nothing-running) prune",
    )
    args = parser.parse_args(argv)

    assert_nothing_running()

    if not args.no_prune:
        result = run(["docker", "image", "prune", "-af"])
        if result.returncode != 0:
            fail(f"prune failed: {result.stderr[-500:]}")
        print(result.stdout.strip().splitlines()[-1] if result.stdout else "pruned")

    headroom = free_gb()
    if headroom < args.min_free_gb:
        fail(f"only {headroom:.1f} GB free after prune (< {args.min_free_gb} GB)")
    print(f"disk headroom: {headroom:.1f} GB (>= {args.min_free_gb} GB required)")

    for task_name in args.prepull:
        image = resolve_image(task_name)
        pull_serialized(image, args.min_free_gb)
        if args.verify_start:
            verify_start(image)
            result = run(["docker", "rmi", image])
            if result.returncode != 0:
                fail(f"could not remove {image} after verification")
            print(f"removed {image} (disk returned: {free_gb():.1f} GB free)")

    print("preflight OK")


if __name__ == "__main__":
    main(sys.argv[1:])
