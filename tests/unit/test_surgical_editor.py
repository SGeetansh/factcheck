"""Comprehensive tests for deterministic surgical patch execution."""

import pytest
from pydantic import ValidationError

from article_rectifier.editing.diff_guard import apply_resolved_edits, replace_span
from article_rectifier.editing.span_resolver import resolve_edit
from article_rectifier.editing.surgical_editor import execute_edits
from article_rectifier.models import (
    EditOperation,
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

ORIGINAL = "Place: Karnataka. Count: 50. Unicode: café."
SOURCE = "The event occurred in West Bengal with 19 attendees."


def sentences() -> list[SentenceSpan]:
    """Create exact immutable spans for the test article.

    Returns:
        Ordered sentence spans.
    """
    return [
        SentenceSpan(
            sentence_id="s1",
            text="Place: Karnataka.",
            start=0,
            end=17,
            paragraph_id="p1",
        ),
        SentenceSpan(
            sentence_id="s2",
            text="Count: 50.",
            start=18,
            end=28,
            paragraph_id="p1",
        ),
        SentenceSpan(
            sentence_id="s3",
            text="Unicode: café.",
            start=29,
            end=len(ORIGINAL),
            paragraph_id="p1",
        ),
    ]


def evidence() -> EvidenceSpan:
    """Create one exact source evidence fixture.

    Returns:
        Verbatim evidence over the complete test source.
    """
    return EvidenceSpan(
        evidence_id="ev1",
        source_article_id="article-1",
        text=SOURCE,
        start=0,
        end=len(SOURCE),
        retrieval_score=1.0,
        retrieval_strategy=RetrievalStrategy.BM25,
    )


def verification(request_id: str, error_type: ErrorType) -> VerificationResult:
    """Create one accepted contradiction fixture.

    Args:
        request_id: Request identity authorized for editing.
        error_type: Verified error classification.

    Returns:
        Actionable source-grounded verification.
    """
    return VerificationResult(
        request_id=request_id,
        verdict=Verdict.CONTRADICTED,
        evidence_ids=("ev1",),
        explanation="The source gives the replacement.",
        error_types=(error_type,),
        proposed_factual_correction="source fact",
        status=VerificationStatus.CORRECTION_AVAILABLE,
    )


def edit(
    patch_id: str = "p1",
    request_id: str = "r1",
    sentence_id: str = "s1",
    original: str = "Karnataka",
    replacement: str = "West Bengal",
    **updates: object,
) -> SurgicalEdit:
    """Create one valid unresolved edit fixture.

    Args:
        patch_id: Stable patch identity.
        request_id: Verification request identity.
        sentence_id: Permitted sentence identity.
        original: Exact target text.
        replacement: Replacement text.
        **updates: Additional edit field overrides.

    Returns:
        Validated surgical edit.
    """
    values: dict[str, object] = {
        "patch_id": patch_id,
        "request_id": request_id,
        "sentence_id": sentence_id,
        "original_text": original,
        "replacement_text": replacement,
        "evidence_ids": ("ev1",),
        "error_type": ErrorType.GEOGRAPHIC,
        "explanation": "The source supports the replacement.",
    }
    values.update(updates)
    return SurgicalEdit(**values)


def execute(proposed: list[SurgicalEdit]):
    """Execute test patches with matching evidence and verifications.

    Args:
        proposed: Proposed unresolved edits.

    Returns:
        Deterministic edit execution result.
    """
    verifications = [
        verification(item.request_id, item.error_type) for item in proposed
    ]
    return execute_edits(
        ORIGINAL,
        proposed,
        sentences(),
        verifications,
        [evidence()],
        SOURCE,
    )


def test_word_and_number_replacements_apply_right_to_left() -> None:
    """Apply multiple original-offset replacements without offset drift.

    Returns:
        None.
    """
    proposed = [
        edit(),
        edit(
            patch_id="p2",
            request_id="r2",
            sentence_id="s2",
            original="50",
            replacement="19",
            error_type=ErrorType.NUMERICAL,
        ),
    ]
    result = execute(proposed)
    assert result.text == "Place: West Bengal. Count: 19. Unicode: café."
    assert all(item.status is PatchStatus.ACCEPTED for item in result.results)
    assert [record.patch_id for record in result.audit_records] == ["p1", "p2"]


def test_repeated_phrase_within_sentence_is_rejected() -> None:
    """Reject ambiguous duplicate targets instead of choosing one.

    Returns:
        None.
    """
    text = "Karnataka and Karnataka."
    span = SentenceSpan(
        sentence_id="s1", text=text, start=0, end=len(text), paragraph_id="p1"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        resolve_edit(text, span, edit())


def test_duplicate_entity_in_other_sentence_is_safe() -> None:
    """Resolve a target uniquely inside its permitted sentence region.

    Returns:
        None.
    """
    text = "Karnataka. Karnataka."
    span = SentenceSpan(
        sentence_id="s2", text="Karnataka.", start=11, end=21, paragraph_id="p1"
    )
    resolved = resolve_edit(text, span, edit(sentence_id="s2"))
    assert resolved.global_start == 11


def test_unicode_offsets_and_newline_replacement_are_exact() -> None:
    """Use Python Unicode indexes and permit intentional replacement newlines.

    Returns:
        None.
    """
    unicode_edit = edit(
        sentence_id="s3",
        original="café",
        replacement="coffee\nshop",
        error_type=ErrorType.NAMED_ENTITY,
    )
    result = execute([unicode_edit])
    assert result.text.endswith("Unicode: coffee\nshop.")


def test_insertion_and_deletion_are_explicit() -> None:
    """Apply anchored zero-length insertion and intentional deletion.

    Returns:
        None.
    """
    insertion = edit(
        operation=EditOperation.INSERT,
        original="",
        replacement="reported ",
        insertion_anchor="Karnataka",
    )
    deletion = edit(
        patch_id="p2",
        request_id="r2",
        operation=EditOperation.DELETE,
        original="Karnataka",
        replacement="",
    )
    inserted = execute([insertion])
    assert inserted.text.startswith("Place: reported Karnataka")
    deleted = execute([deletion])
    assert deleted.text.startswith("Place: .")


def test_overlapping_edits_are_both_rejected_and_article_is_unchanged() -> None:
    """Reject conflicts rather than selecting a patch arbitrarily.

    Returns:
        None.
    """
    proposed = [
        edit(original="Karnataka", replacement="West Bengal"),
        edit(
            patch_id="p2",
            request_id="r2",
            original="Place: Karnataka",
            replacement="Location: West Bengal",
        ),
    ]
    result = execute(proposed)
    assert result.text == ORIGINAL
    assert all(item.status is PatchStatus.REJECTED for item in result.results)


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [(-1, 1, "x"), (0, 99, "x"), (2, 1, "")],
)
def test_replace_span_rejects_invalid_offsets(
    start: int, end: int, expected: str
) -> None:
    """Reject out-of-bounds and reversed offsets.

    Args:
        start: Invalid start offset.
        end: Invalid end offset.
        expected: Expected source span.

    Returns:
        None.
    """
    with pytest.raises(ValueError, match="offset"):
        replace_span("text", start, end, expected, "new")


def test_missing_evidence_reference_rejects_patch() -> None:
    """Keep the article unchanged when evidence authorization fails.

    Returns:
        None.
    """
    proposed = edit(evidence_ids=("missing",))
    result = execute_edits(
        ORIGINAL,
        [proposed],
        sentences(),
        [verification("r1", ErrorType.GEOGRAPHIC)],
        [evidence()],
        SOURCE,
    )
    assert result.text == ORIGINAL
    assert result.results[0].status is PatchStatus.REJECTED


def test_empty_and_no_op_edits_fail_schema_validation() -> None:
    """Prevent accidental empty replacements and no-op patches before execution.

    Returns:
        None.
    """
    with pytest.raises(ValidationError):
        edit(replacement="")
    with pytest.raises(ValidationError):
        edit(replacement="Karnataka")


def test_same_patch_set_is_deterministic() -> None:
    """Produce identical output for repeated execution against one original.

    Returns:
        None.
    """
    proposed = [edit()]
    assert execute(proposed).text == execute(proposed).text


def test_apply_resolved_edit_checks_original_span() -> None:
    """Reject a resolved edit when its expected text does not match the article.

    Returns:
        None.
    """
    resolved = edit().model_copy(update={"global_start": 0, "global_end": 9})
    with pytest.raises(ValueError, match="does not match"):
        apply_resolved_edits(ORIGINAL, [resolved])
