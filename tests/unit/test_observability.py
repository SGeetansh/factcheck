"""Tests for local structured auditing and observed usage accounting."""

import json
from pathlib import Path

from article_rectifier.llm import LLMCallRecord, LLMResponseStatus
from article_rectifier.observability import AuditLogger, summarize_usage


def record(cost: float | None) -> LLMCallRecord:
    """Create one usage record fixture.

    Args:
        cost: Optional provider-reported cost.

    Returns:
        Validated logical call record.
    """
    return LLMCallRecord(
        prompt_version="v1",
        schema_name="Schema",
        schema_version="1",
        input_fingerprint="fingerprint",
        model_identifier="model",
        status=LLMResponseStatus.SUCCESS,
        input_tokens=10,
        output_tokens=4,
        latency_seconds=0.25,
        retry_count=1,
        provider_cost_usd=cost,
    )


def test_audit_logger_writes_compact_valid_jsonl(tmp_path: Path) -> None:
    """Write required event fields without secrets.

    Args:
        tmp_path: Temporary audit directory.

    Returns:
        None.
    """
    destination = tmp_path / "audit.jsonl"
    logger = AuditLogger(destination)
    logger.emit(
        run_id="run-1",
        article_id="article-1",
        batch_id="b0001",
        node_name="planner",
        event_type="facts_planned",
        status="completed",
        duration_seconds=0.2,
        details={"fact_count": 2},
    )
    data = json.loads(destination.read_text(encoding="utf-8"))
    assert data["run_id"] == "run-1"
    assert data["details"] == {"fact_count": 2}
    assert "api_key" not in data


def test_usage_summary_distinguishes_missing_provider_cost() -> None:
    """Do not convert absent billing metadata into a zero-cost claim.

    Returns:
        None.
    """
    missing = summarize_usage([record(None)])
    observed = summarize_usage([record(0.01), record(None)])
    assert missing.provider_cost_usd is None
    assert missing.estimated_cost_usd is None
    assert observed.provider_cost_usd == 0.01
    assert observed.provider_cost_records == 1
    assert observed.llm_calls == 2
    assert observed.input_tokens == 20
    assert observed.output_tokens == 8
    assert observed.retries == 2
