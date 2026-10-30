"""Tests for independent structural, factual, and preservation validation."""

from article_rectifier.editing.surgical_editor import execute_edits
from article_rectifier.models import (
    ErrorType,
    EvidenceSpan,
    PatchStatus,
    RetrievalStrategy,
    SentenceSpan,
    SurgicalEdit,
    Verdict,
    VerificationResult,
    VerificationStatus,
)
from article_rectifier.nodes.validator import validate_patches

ORIGINAL = "The film was blocked in Karnataka."
SOURCE = "Most theaters blocked the film in West Bengal."
SENTENCE = SentenceSpan(
    sentence_id="s1",
    text=ORIGINAL,
    start=0,
    end=len(ORIGINAL),
    paragraph_id="p1",
)
EVIDENCE = EvidenceSpan(
    evidence_id="ev1",
    source_article_id="article-1",
    text=SOURCE,
    start=0,
    end=len(SOURCE),
    retrieval_score=1.0,
    retrieval_strategy=RetrievalStrategy.BM25,
)
VERIFICATION = VerificationResult(
    request_id="r1",
    verdict=Verdict.CONTRADICTED,
    evidence_ids=("ev1",),
    explanation="The source identifies West Bengal.",
    error_types=(ErrorType.GEOGRAPHIC,),
    proposed_factual_correction="West Bengal",
    status=VerificationStatus.CORRECTION_AVAILABLE,
)


def edit(replacement: str = "West Bengal", **updates: object) -> SurgicalEdit:
    """Create one replacement fixture.

    Args:
        replacement: Proposed replacement phrase.
        **updates: Edit field overrides.

    Returns:
        Validated unresolved edit.
    """
    values: dict[str, object] = {
        "patch_id": "p1",
        "request_id": "r1",
        "sentence_id": "s1",
        "original_text": "Karnataka",
        "replacement_text": replacement,
        "evidence_ids": ("ev1",),
        "error_type": ErrorType.GEOGRAPHIC,
        "explanation": "The source gives West Bengal.",
    }
    values.update(updates)
    return SurgicalEdit(**values)


def execution_for(proposed: SurgicalEdit):
    """Run deterministic editing for one proposed patch.

    Args:
        proposed: Unresolved proposed edit.

    Returns:
        Provisional deterministic editor output.
    """
    return execute_edits(
        ORIGINAL,
        [proposed],
        [SENTENCE],
        [VERIFICATION],
        [EVIDENCE],
        SOURCE,
    )


def validate(execution, **updates: object):
    """Validate one editor execution with standard fixtures.

    Args:
        execution: Provisional editor output.
        **updates: Validator option overrides.

    Returns:
        Final validation execution.
    """
    values: dict[str, object] = {
        "original_text": ORIGINAL,
        "candidate_text": execution.text,
        "patch_results": list(execution.results),
        "sentences": [SENTENCE],
        "evidence": [EVIDENCE],
        "source_text": SOURCE,
    }
    values.update(updates)
    return validate_patches(**values)


def test_grounded_patch_is_accepted() -> None:
    """Accept a reconstructed patch whose replacement occurs in evidence.

    Returns:
        None.
    """
    result = validate(execution_for(edit()))
    assert result.text == "The film was blocked in West Bengal."
    assert result.results[0].status is PatchStatus.ACCEPTED


def test_unauthorized_candidate_change_rejects_and_restores_original() -> None:
    """Reject output that differs from exact patch reconstruction.

    Returns:
        None.
    """
    execution = execution_for(edit())
    result = validate(
        execution,
        candidate_text="A rewritten film sentence in West Bengal.",
    )
    assert result.text == ORIGINAL
    assert result.results[0].status is PatchStatus.REJECTED


def test_unsupported_replacement_is_retryable_then_unresolved() -> None:
    """Roll back a replacement absent from evidence with a bounded retry.

    Returns:
        None.
    """
    execution = execution_for(edit("Kerala"))
    first = validate(execution, retry_count=0, max_patch_retries=1)
    final = validate(execution, retry_count=1, max_patch_retries=1)
    assert first.text == ORIGINAL
    assert first.results[0].status is PatchStatus.RETRYABLE
    assert final.results[0].status is PatchStatus.UNRESOLVED


def test_missing_evidence_rolls_back_patch() -> None:
    """Do not retain an edit when cited evidence is absent at validation.

    Returns:
        None.
    """
    execution = execution_for(edit())
    result = validate(execution, evidence=[])
    assert result.text == ORIGINAL
    assert result.results[0].status is PatchStatus.RETRYABLE


def test_rejected_editor_patch_never_changes_article() -> None:
    """Preserve original text for a patch rejected before validation.

    Returns:
        None.
    """
    bad = edit(evidence_ids=("missing",))
    execution = execute_edits(
        ORIGINAL,
        [bad],
        [SENTENCE],
        [VERIFICATION],
        [EVIDENCE],
        SOURCE,
    )
    result = validate(execution)
    assert result.text == ORIGINAL
    assert result.results[0].status is PatchStatus.REJECTED


def test_large_grounded_patch_warns_but_is_not_automatically_rejected() -> None:
    """Treat the preservation threshold as a warning rather than a hard cap.

    Returns:
        None.
    """
    execution = execution_for(edit())
    result = validate(execution, soft_patch_character_limit=5)
    assert result.results[0].status is PatchStatus.ACCEPTED
    assert result.warnings == ("large_patch:p1",)


def test_deletion_without_exact_corrected_sentence_support_is_unresolved() -> None:
    """Fail closed when deterministic checks cannot establish a safe deletion.

    Returns:
        None.
    """
    deletion = edit(
        replacement="",
        operation="delete",
        original_text="Karnataka",
    )
    execution = execution_for(deletion)
    result = validate(execution, retry_count=1, max_patch_retries=1)
    assert result.text == ORIGINAL
    assert result.results[0].status is PatchStatus.UNRESOLVED
