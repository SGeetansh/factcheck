"""Mocked integration tests for LangGraph and article-level reconstruction."""

import json
from pathlib import Path

from article_rectifier.config import Settings
from article_rectifier.llm import LLMClient, MockLLMTransport
from article_rectifier.models import ArticleInput, ErrorType
from article_rectifier.observability import AuditLogger
from article_rectifier.pipeline import (
    ArticleProcessingStatus,
    ArticleRectificationPipeline,
)
from article_rectifier.preprocessing.sentence_splitter import split_sentences

ARTICLE_ID = "article-test"
GENERATED = "The film was blocked in Karnataka. Attendance was 50."
SOURCE = "Most theaters blocked the film in West Bengal.\n\nAttendance was 19."


def response(value: object) -> dict[str, object]:
    """Build a LiteLLM-shaped response containing JSON.

    Args:
        value: JSON-compatible structured result or raw string.

    Returns:
        Mock transport response.
    """
    content = value if isinstance(value, str) else json.dumps(value)
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }


def plan_for_generated() -> dict[str, object]:
    """Create a two-request plan using exact generated sentence offsets.

    Returns:
        Schema-valid fact-check plan mapping.
    """
    first, second = split_sentences(GENERATED)
    return {
        "requests": [
            {
                "request_id": "req-place",
                "sentence_id": first.sentence_id,
                "claim_text": first.text,
                "verification_question": "Where was the film blocked?",
                "search_query": "film blocked theaters state",
                "error_category_hints": ["geographic"],
                "sentence_start": first.start,
                "sentence_end": first.end,
            },
            {
                "request_id": "req-count",
                "sentence_id": second.sentence_id,
                "claim_text": second.text,
                "verification_question": "What was attendance?",
                "search_query": "attendance count",
                "error_category_hints": ["numerical"],
                "sentence_start": second.start,
                "sentence_end": second.end,
            },
        ]
    }


def verification(
    request_id: str,
    evidence_id: str,
    correction: str,
    error_type: ErrorType,
) -> dict[str, object]:
    """Create one actionable verification response.

    Args:
        request_id: Planned request identity.
        evidence_id: Stable exact source passage identity.
        correction: Correct fact present in the passage.
        error_type: Fixed factual error category.

    Returns:
        Schema-valid contradiction mapping.
    """
    return {
        "request_id": request_id,
        "verdict": "contradicted",
        "evidence_ids": [evidence_id],
        "explanation": "The exact source passage establishes the correction.",
        "error_types": [error_type.value],
        "proposed_factual_correction": correction,
        "status": "correction_available",
    }


def patch(
    patch_id: str,
    request_id: str,
    sentence_id: str,
    old: str,
    new: str,
    evidence_id: str,
    error_type: ErrorType,
) -> dict[str, object]:
    """Create one minimal replacement proposal response.

    Args:
        patch_id: Stable patch identity.
        request_id: Verified request identity.
        sentence_id: Permitted sentence identity.
        old: Exact generated phrase.
        new: Source-grounded replacement.
        evidence_id: Cited source passage identity.
        error_type: Verified factual category.

    Returns:
        Patch proposal envelope.
    """
    return {
        "edits": [
            {
                "patch_id": patch_id,
                "request_id": request_id,
                "sentence_id": sentence_id,
                "operation": "replace",
                "original_text": old,
                "replacement_text": new,
                "evidence_ids": [evidence_id],
                "error_type": error_type.value,
                "explanation": "The source contains the replacement value.",
            }
        ]
    }


def pipeline(
    tmp_path: Path,
    outcomes: list[object],
    *,
    batch_size: int = 3,
) -> tuple[ArticleRectificationPipeline, MockLLMTransport]:
    """Create a pipeline with scripted offline model outcomes.

    Args:
        tmp_path: Temporary project root.
        outcomes: Ordered mock response mappings.
        batch_size: Editable sentences per batch.

    Returns:
        Pipeline and inspectable transport.
    """
    transport = MockLLMTransport(outcomes)
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        max_llm_retries=0,
        batch_size=batch_size,
        enable_validation_llm=False,
        _env_file=None,
    )
    return (
        ArticleRectificationPipeline(settings, LLMClient(settings, transport)),
        transport,
    )


def two_error_outcomes() -> list[object]:
    """Create all five model responses for a two-error article.

    Returns:
        Planner, verifier, and proposer responses in workflow order.
    """
    first, second = split_sentences(GENERATED)
    return [
        response(plan_for_generated()),
        response(
            verification(
                "req-place",
                "src-article-test-0001",
                "West Bengal",
                ErrorType.GEOGRAPHIC,
            )
        ),
        response(
            verification(
                "req-count",
                "src-article-test-0003",
                "19",
                ErrorType.NUMERICAL,
            )
        ),
        response(
            patch(
                "patch-place",
                "req-place",
                first.sentence_id,
                "Karnataka",
                "West Bengal",
                "src-article-test-0001",
                ErrorType.GEOGRAPHIC,
            )
        ),
        response(
            patch(
                "patch-count",
                "req-count",
                second.sentence_id,
                "50",
                "19",
                "src-article-test-0003",
                ErrorType.NUMERICAL,
            )
        ),
    ]


def test_graph_corrects_two_errors_and_preserves_other_text(tmp_path: Path) -> None:
    """Run all explicit nodes and merge two original-offset patches.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    service, transport = pipeline(tmp_path, two_error_outcomes())
    result = service.rectify(
        ArticleInput(
            article_id=ARTICLE_ID, source_text=SOURCE, generated_text=GENERATED
        )
    )
    assert (
        result.output_text == "The film was blocked in West Bengal. Attendance was 19."
    )
    assert result.status is ArticleProcessingStatus.COMPLETED
    assert len(result.accepted_edits) == 2
    assert len(result.llm_records) == 5
    assert len(transport.calls) == 5


def test_article_with_no_planned_facts_remains_exact(tmp_path: Path) -> None:
    """Preserve an unchanged article when the planner returns no claims.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    generated = "A quiet essay without concrete claims."
    service, transport = pipeline(tmp_path, [response({"requests": []})])
    result = service.rectify(
        ArticleInput(
            article_id=ARTICLE_ID, source_text="A source.", generated_text=generated
        )
    )
    assert result.output_text == generated
    assert result.status is ArticleProcessingStatus.COMPLETED
    assert len(transport.calls) == 1


def test_failed_planner_call_uses_safe_degraded_fallback(tmp_path: Path) -> None:
    """Keep sanitized original text after repeated malformed planner output.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    generated = "A quiet essay.\n\n**Error Annotations:**\n[]"
    service, _ = pipeline(tmp_path, [response("bad"), response("still bad")])
    result = service.rectify(
        ArticleInput(
            article_id=ARTICLE_ID, source_text="A source.", generated_text=generated
        )
    )
    assert result.output_text == "A quiet essay.\n\n"
    assert result.status is ArticleProcessingStatus.DEGRADED
    assert result.errors


def test_multiple_batches_are_processed_sequentially(tmp_path: Path) -> None:
    """Invoke one planner per batch and preserve complete reconstruction.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    generated = "One claim. Two claim. Three claim. Four claim."
    outcomes = [response({"requests": []}), response({"requests": []})]
    service, transport = pipeline(tmp_path, outcomes, batch_size=3)
    result = service.rectify(
        ArticleInput(
            article_id=ARTICLE_ID, source_text="Source text.", generated_text=generated
        )
    )
    assert result.output_text == generated
    assert len(transport.calls) == 2


def test_validation_failure_gets_one_targeted_retry(tmp_path: Path) -> None:
    """Retry one unsupported replacement and accept the grounded correction.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    first = split_sentences(GENERATED)[0]
    plan = plan_for_generated()
    plan["requests"] = [plan["requests"][0]]
    outcomes = [
        response(plan),
        response(
            verification(
                "req-place",
                "src-article-test-0001",
                "West Bengal",
                ErrorType.GEOGRAPHIC,
            )
        ),
        response(
            patch(
                "patch-place",
                "req-place",
                first.sentence_id,
                "Karnataka",
                "Kerala",
                "src-article-test-0001",
                ErrorType.GEOGRAPHIC,
            )
        ),
        response(
            patch(
                "patch-place-retry",
                "req-place",
                first.sentence_id,
                "Karnataka",
                "West Bengal",
                "src-article-test-0001",
                ErrorType.GEOGRAPHIC,
            )
        ),
    ]
    service, transport = pipeline(tmp_path, outcomes)
    retry_generated = first.text
    retry_source = SOURCE.split("\n\n", maxsplit=1)[0]
    result = service.rectify(
        ArticleInput(
            article_id=ARTICLE_ID,
            source_text=retry_source,
            generated_text=retry_generated,
        )
    )
    assert result.output_text.startswith("The film was blocked in West Bengal.")
    assert len(transport.calls) == 4
    retry_messages = str(transport.calls[3]["messages"])
    assert "replacement value is absent" in retry_messages


def test_pipeline_emits_required_local_audit_events(tmp_path: Path) -> None:
    """Record article, batch, and explicit node lifecycle events as JSONL.

    Args:
        tmp_path: Temporary project and audit root.

    Returns:
        None.
    """
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        max_llm_retries=0,
        _env_file=None,
    )
    transport = MockLLMTransport([response({"requests": []})])
    audit_path = tmp_path / "events.jsonl"
    service = ArticleRectificationPipeline(
        settings,
        LLMClient(settings, transport),
        AuditLogger(audit_path),
    )
    service.rectify(
        ArticleInput(
            article_id=ARTICLE_ID,
            source_text="A source.",
            generated_text="A quiet essay.",
        )
    )
    events = [
        json.loads(line)["event_type"] for line in audit_path.read_text().splitlines()
    ]
    assert events == [
        "article_started",
        "batch_created",
        "facts_planned",
        "evidence_retrieved",
        "claim_verified",
        "patch_proposed",
        "patch_skipped",
        "batch_completed",
        "article_completed",
    ]


def test_duplicate_detections_share_one_coordinated_patch_call(tmp_path: Path) -> None:
    """Resolve agreeing detections once instead of rejecting overlapping edits.

    Args:
        tmp_path: Isolated cache and diagnostic directory.
    """
    original = "The film was blocked in Karnataka."
    request = plan_for_generated()["requests"][0]
    second = {**request, "request_id": "req-place-2", "claim_text": "Karnataka"}
    evidence_id = "src-article-test-0001"
    outcomes = [
        response({"requests": [request, second]}),
        response(
            verification("req-place", evidence_id, "West Bengal", ErrorType.GEOGRAPHIC)
        ),
        response(
            verification(
                "req-place-2", evidence_id, "West Bengal", ErrorType.GEOGRAPHIC
            )
        ),
        response(
            patch(
                "p1",
                "req-place",
                "s0001",
                "Karnataka",
                "West Bengal",
                evidence_id,
                ErrorType.GEOGRAPHIC,
            )
        ),
    ]
    service, transport = pipeline(tmp_path, outcomes)
    result = service.rectify(
        ArticleInput(
            article_id=ARTICLE_ID,
            source_text="Most theaters blocked the film in West Bengal.",
            generated_text=original,
        )
    )
    assert result.output_text == "The film was blocked in West Bengal."
    assert len(result.accepted_edits) == 1
    assert not result.rejected_patches
    assert len(transport.calls) == 4
    task = json.loads(transport.calls[-1]["messages"][-1]["content"])
    assert len(task["verified_issues"]) == 2


def test_semantic_rejection_is_bounded_and_preserves_other_accepted_edits(
    tmp_path: Path,
) -> None:
    """A failed sentence review must not discard an accepted edit elsewhere.

    Args:
        tmp_path: Isolated runtime directory.
    """
    uncertain = response(
        {
            "verdict": "uncertain",
            "evidence_ids": ["src-article-test-0001"],
            "explanation": "The edited relationship is not established.",
        }
    )
    supported = response(
        {
            "verdict": "supported",
            "evidence_ids": ["src-article-test-0003"],
            "explanation": "The attendance correction matches the source.",
        }
    )
    initial = two_error_outcomes()
    service, transport = pipeline(
        tmp_path, [*initial, uncertain, supported, initial[3], uncertain, supported]
    )
    service.settings.enable_validation_llm = True
    result = service.rectify(
        ArticleInput(
            article_id=ARTICLE_ID, source_text=SOURCE, generated_text=GENERATED
        )
    )
    assert result.output_text == "The film was blocked in Karnataka. Attendance was 19."
    assert len(result.accepted_edits) == 1
    assert len(result.rejected_patches) == 1
    assert result.rejected_patches[0].status.value == "unresolved"
    assert len(transport.calls) == 10
