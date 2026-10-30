"""Stable serialization for evaluation reports."""

from pathlib import Path

from evaluate.models import EvaluationReport


def write_report(report: EvaluationReport, destination: Path) -> None:
    """Write one evaluation report as formatted JSON.

    Args:
        report: Validated evaluation report.
        destination: Output JSON file.

    Returns:
        None.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
