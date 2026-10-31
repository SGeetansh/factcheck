"""Print proposed edits and every patch decision from an existing JSONL trace.

This tool only reads local logs. It never calls the LLM API.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_trace(path: Path) -> list[dict[str, Any]]:
    """Read JSONL events in chronological order.

    Args:
        path: Existing debug trace file.

    Returns:
        Parsed events in the order they were recorded.
    """
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def show_patch(
    location: str,
    stage: str,
    edit: dict[str, Any],
    status: str,
    reason: str | None = None,
) -> None:
    """Print the exact edit and its current disposition.

    Args:
        location: Article and batch identifier.
        stage: Stage responsible for the current decision.
        edit: Serialized surgical edit model.
        status: Proposed, accepted, rejected, retryable, or unresolved.
        reason: Optional rejection or retry explanation.

    Returns:
        None.
    """
    print(f"\n[{location}] {stage}: {status.upper()} ({edit.get('patch_id', '?')})")
    print(f"  sentence: {edit.get('sentence_id', '?')}")
    print(f"  before: {edit.get('original_text', '')!r}")
    print(f"  after:  {edit.get('replacement_text', '')!r}")
    print(f"  evidence: {', '.join(edit.get('evidence_ids', []))}")
    if reason:
        print(f"  WHY: {reason}")


def show_proposals(location: str, details: dict[str, Any]) -> None:
    """Print raw model edits, including those rejected by the proposer.

    Args:
        location: Article and batch identifier.
        details: Data from a proposer_output event.

    Returns:
        None.
    """
    raw = (details.get("raw_proposal") or {}).get("edits", [])
    allowed = {
        (edit.get("patch_id"), edit.get("sentence_id"))
        for edit in details.get("accepted_proposals", [])
    }
    codes = details.get("rejected_codes", [])
    for edit in raw:
        accepted = (edit.get("patch_id"), edit.get("sentence_id")) in allowed
        reasons = [code for code in codes if edit.get("patch_id", "") in code]
        show_patch(
            location,
            "PROPOSER",
            edit,
            "passed" if accepted else "rejected",
            "; ".join(reasons) if reasons else None,
        )
    if codes:
        print(f"[{location}] proposer error codes: {codes}")


def show_results(location: str, stage: str, results: list[dict[str, Any]]) -> None:
    """Print executor, validator, or final results for each patch.

    Args:
        location: Article and batch identifier.
        stage: Name of the decision stage.
        results: Serialized PatchResult objects.

    Returns:
        None.
    """
    for item in results:
        show_patch(
            location,
            stage,
            item["edit"],
            item["status"],
            item.get("reason"),
        )


def show_events(
    events: list[dict[str, Any]], *, include_verifier: bool = False
) -> None:
    """Print the patch lifecycle in trace order.

    Args:
        events: All parsed events for one debug run.
        include_verifier: Show actionable verifier results and failures.

    Returns:
        None.
    """
    for event in events:
        stage = event.get("stage")
        details = event.get("details", {})
        location = f"{event.get('article_id')}/{event.get('batch_id')}"
        if stage == "verifier_output" and include_verifier:
            result = details.get("effective_result") or {}
            raw = details.get("raw_result") or {}
            if result.get("verdict") == "contradicted" or details.get("rejected_codes"):
                print(
                    f"[{location}] VERIFIER request={details.get('request_id')} "
                    f"raw={raw.get('verdict', '?')} "
                    f"final={result.get('verdict', '?')} "
                    f"status={result.get('status', '?')} "
                    f"correction={result.get('proposed_factual_correction')!r} "
                    f"errors={details.get('rejected_codes', [])}"
                )
        elif stage == "proposer_output":
            show_proposals(location, details)
        elif stage == "executor_output":
            show_results(location, "EXECUTOR", details.get("patch_results", []))
        elif stage == "validator_output":
            show_results(location, "VALIDATOR", details.get("after", []))
        elif stage == "sentence_review":
            review = details.get("review") or {}
            print(
                f"[{location}] SENTENCE REVIEW: "
                f"{'ACCEPTED' if details.get('accepted') else 'REJECTED'} "
                f"{details.get('sentence_id')} "
                f"verdict={review.get('verdict', 'no_response')} "
                f"reason={review.get('explanation', 'none')}"
            )
        elif stage == "batch_finished":
            show_results(location, "FINAL BATCH", details.get("patch_results", []))


def main() -> None:
    """Parse a trace path and print local patch decisions.

    Returns:
        None.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Path to a debug JSONL trace")
    parser.add_argument(
        "--include-verifier",
        action="store_true",
        help="Also show actionable factual verification results",
    )
    args = parser.parse_args()
    show_events(read_trace(args.trace), include_verifier=args.include_verifier)


if __name__ == "__main__":
    main()
