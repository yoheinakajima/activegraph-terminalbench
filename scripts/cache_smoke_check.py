#!/usr/bin/env python3
"""Assert prompt caching worked on a smoke-test trial.

Usage:
    python scripts/cache_smoke_check.py <trial_dir>/agent [--base-input-price 1.0]

Reads event_log.json, prints a per-turn cache table, and asserts the v1
caching invariants:

1. cache_read_input_tokens > 0 from turn 3 through the last turn before
   the transcript window starts sliding (v1 keeps MAX_TAIL_EXCHANGES
   verbatim exchanges; once trimming starts the prefix changes every turn
   and can never hit, so the harness deliberately stops paying for cache
   writes there).
2. Post-trim turns write (almost) nothing to cache: a guaranteed-miss
   write would bill 1.25x for entries nothing ever reads.
3. The effective input cost is below the uncached equivalent.

Also prints effective input cost vs the uncached equivalent. Prices are
per MTok for the model under test (haiku 4-5 default: $1 input; 5-minute
cache writes bill 1.25x input, cache reads 0.1x).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from activegraph_harness.context import MAX_TAIL_EXCHANGES

CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1
ASSERT_FROM_TURN = 3  # 1-based
# Turn N's request contains N-1 prior exchanges; the window slides (and
# caching intentionally stops) once those exceed MAX_TAIL_EXCHANGES.
LAST_CACHED_TURN = MAX_TAIL_EXCHANGES + 1
# Post-trim writes should be zero; allow a little noise for the system
# block in case the model's minimum cacheable prefix is small.
POST_TRIM_WRITE_TOLERANCE = 2000


def main(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("agent_dir", type=Path)
    parser.add_argument("--base-input-price", type=float, default=1.0)
    args = parser.parse_args(argv)

    log = json.loads((args.agent_dir / "event_log.json").read_text())
    turns = [e["payload"] for e in log["events"] if e["type"] == "model_turn"]
    if not turns:
        raise SystemExit("no model_turn events found")

    print(f"{'turn':>4} {'uncached_in':>12} {'cache_write':>12} {'cache_read':>12}")
    failures = []
    for i, t in enumerate(turns, start=1):
        read = t.get("cache_read_input_tokens")
        write = t.get("cache_creation_input_tokens")
        if read is None or write is None:
            raise SystemExit(f"turn {i} is missing cache fields; wrong build?")
        print(f"{i:>4} {t['input_tokens']:>12} {write:>12} {read:>12}")
        if ASSERT_FROM_TURN <= i <= LAST_CACHED_TURN and read <= 0:
            failures.append(f"turn {i}: no cache read inside the stable window")
        if i > LAST_CACHED_TURN and write > POST_TRIM_WRITE_TOLERANCE:
            failures.append(
                f"turn {i}: {write} wasted cache-write tokens after the "
                "window started sliding"
            )

    uncached_in = sum(t["input_tokens"] for t in turns)
    writes = sum(t["cache_creation_input_tokens"] for t in turns)
    reads = sum(t["cache_read_input_tokens"] for t in turns)
    total_prompt = uncached_in + writes + reads

    price = args.base_input_price / 1_000_000
    effective = (
        uncached_in * price
        + writes * price * CACHE_WRITE_MULTIPLIER
        + reads * price * CACHE_READ_MULTIPLIER
    )
    baseline = total_prompt * price
    hit_rate = reads / total_prompt if total_prompt else 0.0

    print(f"\ntotal prompt tokens: {total_prompt:,}")
    print(f"  uncached suffix:   {uncached_in:,}")
    print(f"  cache writes:      {writes:,}")
    print(f"  cache reads:       {reads:,}  (hit rate {hit_rate:.1%})")
    print(f"effective input cost: ${effective:.4f}")
    print(f"uncached would cost:  ${baseline:.4f}")
    if baseline > 0:
        print(f"input cost ratio:     {effective / baseline:.2f}x")

    summary = json.loads((args.agent_dir / "summary.json").read_text())
    for key in ("tokens_cache_read", "tokens_cache_creation", "cache_hit_rate"):
        if key not in summary:
            raise SystemExit(f"summary.json is missing {key}")
    print(f"summary.json cache_hit_rate: {summary['cache_hit_rate']}")

    if baseline > 0 and effective >= baseline:
        failures.append(
            f"effective input cost ${effective:.4f} is not below the "
            f"uncached ${baseline:.4f}"
        )
    if failures:
        raise SystemExit("FAIL:\n  " + "\n  ".join(failures))
    print(
        f"PASS: cache reads present turns {ASSERT_FROM_TURN}..."
        f"{min(LAST_CACHED_TURN, len(turns))}, no wasted post-trim writes, "
        "effective cost below uncached"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
