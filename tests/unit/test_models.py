"""Schema tests for strict rectification data contracts."""

import pytest
from pydantic import ValidationError

from article_rectifier.models import (
    EditOperation,
    ErrorType,
    EvidenceSpan,
    FactCheckPlan,
    FactCheckRequest,
    PatchResult,
    PatchStatus,
    RetrievalStrategy,
    SentenceSpan,
    SurgicalEdit,
    Verdict,
    VerificationResult,
    VerificationStatus,
)


def make_request() -> FactCheckRequest:
    """Create one valid request fixture.

    Returns:
        A valid fact-check request.
    """
    return FactCheckRequest(
        request_id="req-1",
        sentence_id="s-1",
        claim_text="The count was 50.",
        verification_question="What was the count?",
        search_query="event count",
        error_category_hints=(ErrorType.NUMERICAL,),
        sentence_start=0,
        sentence_end=17,
    )


def make_edit(**updates: object) -> SurgicalEdit:
    """Create one valid edit fixture with optional field overrides.

    Args:
        updates: Model fields to replace.

    Returns:
        A validated surgical edit.
    """
    values: dict[str, object] = {
        "patch_id": "patch-1",
        "request_id": "req-1",
        "sentence_id": "s-1",
        "original_text": "50",
        "replacement_text": "19",
        "evidence_ids": ("ev-1",),
        "error_type": ErrorType.NUMERICAL,
        "explanation": "The source reports 19.",
    }
    values.update(updates)
    return SurgicalEdit(**values)


def test_sentence_offsets_must_match_text() -> None:
    """Reject malformed immutable sentence offsets.

    Returns:
        None.
    """
    with pytest.raises(ValidationError, match="offsets"):
        SentenceSpan(
            sentence_id="s-1",
            text="Text.",
            start=0,
            end=4,
            paragraph_id="p-1",
        )


def test_fact_plan_rejects_duplicate_request_ids() -> None:
    """Prevent ambiguous result-to-request joins.

    Returns:
        None.
    """
    request = make_request()
    with pytest.raises(ValidationError, match="unique"):
        FactCheckPlan(requests=(request, request))


def test_evidence_verifies_exact_source_substring() -> None:
    """Check source evidence against its immutable offsets.

    Returns:
        None.
    """
    evidence = EvidenceSpan(
        evidence_id="ev-1",
        source_article_id="article-1",
        text="19 people",
        start=9,
        end=18,
        retrieval_score=2.0,
        retrieval_strategy=RetrievalStrategy.BM25,
    )
    assert evidence.verify_against("At least 19 people attended.")
    assert not evidence.verify_against("At least 50 people attended.")


@pytest.mark.parametrize(
    "updates",
    [
        {"original_text": "", "replacement_text": "19"},
        {"original_text": "50", "replacement_text": ""},
        {"original_text": "50", "replacement_text": "50"},
        {"global_start": 10, "global_end": None},
    ],
)
def test_replacement_edits_reject_malformed_fields(updates: dict[str, object]) -> None:
    """Reject unsafe replacement proposals.

    Args:
        updates: Invalid field combination.

    Returns:
        None.
    """
    with pytest.raises(ValidationError):
        make_edit(**updates)


def test_explicit_insertion_and_deletion_are_valid() -> None:
    """Allow empty spans only for explicit operations.

    Returns:
        None.
    """
    insertion = make_edit(
        operation=EditOperation.INSERT,
        original_text="",
        replacement_text="not ",
    )
    deletion = make_edit(
        operation=EditOperation.DELETE,
        original_text="allegedly ",
        replacement_text="",
    )
    assert insertion.operation is EditOperation.INSERT
    assert deletion.operation is EditOperation.DELETE


def test_correction_requires_contradiction_and_evidence() -> None:
    """Reject a correction that is not supported by a contradiction verdict.

    Returns:
        None.
    """
    with pytest.raises(ValidationError, match="contradicted"):
        VerificationResult(
            request_id="req-1",
            verdict=Verdict.UNSUPPORTED,
            explanation="The source does not answer the question.",
            proposed_factual_correction="19",
            status=VerificationStatus.CORRECTION_AVAILABLE,
        )


def test_accepted_patch_requires_final_offsets() -> None:
    """Prevent an unresolved proposal from appearing accepted.

    Returns:
        None.
    """
    with pytest.raises(ValidationError, match="offsets"):
        PatchResult(edit=make_edit(), status=PatchStatus.ACCEPTED)


def test_unknown_enum_value_is_rejected() -> None:
    """Reject error categories outside the fixed vocabulary.

    Returns:
        None.
    """
    data = make_request().model_dump(mode="json")
    data["error_category_hints"] = ["stylistic"]
    with pytest.raises(ValidationError):
        FactCheckRequest.model_validate(data)
