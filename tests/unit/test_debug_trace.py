"""Unit tests for local opt-in structured debug traces."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from article_rectifier.observability.debug import DebugTrace


class ExampleResult(BaseModel):
    """Represent one small structured response for trace tests."""

    answer: str


def test_debug_trace_preserves_stage_and_truncates_text(tmp_path: Path) -> None:
    """Ensure node decisions are recorded without unbounded content.

    Args:
        tmp_path: Isolated directory for the debug JSONL file.
    """
    trace = DebugTrace(tmp_path, max_text_chars=8)
    trace.record(
        run_id="run123",
        article_id="article_001",
        batch_id="batch-1",
        stage="verifier_output",
        details={"raw_result": ExampleResult(answer="West Bengal")},
    )
    events = (tmp_path / "run123.jsonl").read_text(encoding="utf-8")
    event = json.loads(events)
    assert event["stage"] == "verifier_output"
    assert event["details"]["raw_result"]["answer"] == "West Ben"
    assert event["article_id"] == "article_001"
