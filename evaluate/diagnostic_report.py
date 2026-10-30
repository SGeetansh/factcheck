"""Reproduce two-article diagnostic metrics from saved outputs and paid traces.

These articles are development cases. Model reviews are not factuality scores.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from article_rectifier.preprocessing.sanitizer import sanitize_generated_article
from evaluate.evaluator import evaluate_article
from evaluate.models import PatchObservation, UsageObservation


def trace_metrics(path: Path) -> tuple[str, dict[str, object], UsageObservation]:
    """Read recorded operations without invoking the rectifier or a model.

    Args:
        path: One complete per-article JSONL trace.

    Returns:
        Article identity, operational observations, and observed token usage.
    """
    events = [json.loads(line) for line in path.read_text().splitlines()]
    batches = [
        event["details"] for event in events if event["stage"] == "batch_finished"
    ]
    records = [record for batch in batches for record in batch["llm_records"]]
    patches = [patch for batch in batches for patch in batch["patch_results"]]
    durations: Counter[str] = Counter()
    for event in events:
        if event["stage"] == "node_finished":
            details = event["details"]
            durations[details["node"]] += details["duration_seconds"]
    operations = {
        "patch_statuses": dict(Counter(patch["status"] for patch in patches)),
        "node_seconds": dict(durations),
        "errors": dict(
            Counter(error for batch in batches for error in batch["errors"])
        ),
        "cache_hits": sum(record["status"] == "cache_hit" for record in records),
        "trace": str(path.resolve()),
        "accepted_patches": [
            patch["edit"] for patch in patches if patch["status"] == "accepted"
        ],
        "final_patch_observations": [PatchObservation(
            patch_id=patch["edit"]["patch_id"], original_text=patch["edit"]["original_text"],
            replacement_text=patch["edit"]["replacement_text"], applied=patch["status"] == "accepted",
            evidence_ids=tuple(patch["edit"]["evidence_ids"]), justified=None,
        ).model_dump(mode="json") for patch in patches],
    }
    usage = UsageObservation(
        llm_calls=sum(record.get("transport_calls") or 0 for record in records),
        input_tokens=sum(record["input_tokens"] for record in records),
        output_tokens=sum(record["output_tokens"] for record in records),
        retries=sum(record["retry_count"] for record in records),
        latency_seconds=sum(record["latency_seconds"] for record in records),
    )
    return events[0]["article_id"], operations, usage


def build_report(project: Path, outputs: Path, traces: list[Path]) -> dict[str, object]:
    """Evaluate saved article outputs against immutable inputs and references.

    Args:
        project: Repository containing original source and generated files.
        outputs: Dedicated output directory from the bounded CLI run.
        traces: Exactly the trace files belonging to that run.

    Returns:
        Independent deterministic metrics and separate operational observations.
    """
    articles = {}
    for trace in traces:
        article_id, operations, usage = trace_metrics(trace)
        filename = f"{article_id}.txt"
        original = sanitize_generated_article(
            (project / "ai_generated_articles" / filename).read_text()
        )
        source = (project / "source_articles" / filename).read_text()
        reference_path = project / "evaluate/fixtures/human_references" / filename
        reference = reference_path.read_text() if reference_path.is_file() else None
        output = (outputs / filename).read_text()
        patches = [PatchObservation.model_validate(item)
                   for item in operations["final_patch_observations"]]
        metrics = evaluate_article(source, original, output, reference, usage=usage,
                                   patches=patches)
        baseline = evaluate_article(source, original, original, reference)
        articles[article_id] = {
            "metrics": metrics.model_dump(mode="json"),
            "operations": operations,
            "unchanged_reference_similarity": baseline.human_reference_similarity.value,
            "output": str((outputs / filename).resolve()),
        }
    return {
        "scope": "Saved diagnostic articles; development results are not holdout estimates.",
        "limitations": "Reference similarity is not accuracy. Attribution, precision, "
        "new-error count and global recall require independent validated judgments. "
        "Operational sentence reviews do not supply those judgments.",
        "articles": articles,
    }


def main() -> None:
    """Write reproducible JSON metrics without network or model access."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--traces", type=Path, nargs="+", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    report = build_report(project, args.outputs, args.traces)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    for article_id, result in report["articles"].items():
        metrics = result["metrics"]
        print(
            article_id,
            "preservation:",
            metrics["preservation"]["value"],
            "reference similarity:",
            metrics["human_reference_similarity"]["value"],
            "patches:",
            result["operations"]["patch_statuses"],
        )


if __name__ == "__main__":
    main()
