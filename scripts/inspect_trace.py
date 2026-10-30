"""Summarize one local rectification debug trace without API calls."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_events(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL trace in recorded order.

    Args:
        path: File written by DebugTrace.

    Returns:
        All well-formed trace events.
    """
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def summarize(events: list[dict[str, Any]]) -> None:
    """Print stage counts, patch failures, and verifier disagreement.

    Args:
        events: Debug trace event mappings in recorded order.
    """
    stages: Counter[str] = Counter()
    failures: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    for event in events:
        stage = event["stage"]
        details = event.get("details", {})
        stages[stage] += 1
        if stage == "node_finished":
            durations[details["node"]] += details["duration_seconds"]
        failures.update(details.get("rejected_codes", []))
        if stage == "executor_output":
            for item in details.get("patch_results", []):
                statuses[item["status"]] += 1
                if item.get("reason"):
                    failures[item["reason"]] += 1
        elif stage == "validator_output":
            before = {item["edit"]["patch_id"]: item for item in details["before"]}
            for item in details["after"]:
                earlier = before.get(item["edit"]["patch_id"], {})
                if item["status"] != earlier.get("status"):
                    failures[item.get("reason") or "validator_changed_status"] += 1
    print("Stage events:")
    for name, count in sorted(stages.items()):
        print(f"  {name}: {count}")
    print("Executor patch outcomes:", dict(statuses))
    print("Node wall time (seconds):")
    for name, duration in durations.most_common():
        print(f"  {name}: {duration:.3f}")
    print("Most common rejection reasons:")
    for reason, count in failures.most_common(20):
        print(f"  {count:3}  {reason}")


def main() -> None:
    """Read a specified trace and optionally print full stage decisions."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to one run JSONL trace")
    parser.add_argument("--details", action="store_true", help="Print full events")
    args = parser.parse_args()
    events = read_events(args.trace)
    summarize(events)
    if args.details:
        for event in events:
            print(json.dumps(event, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
