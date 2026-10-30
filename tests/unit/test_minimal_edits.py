"""Check exact contextual resolution, independent replacements, and retry retention."""

from article_rectifier.editing.diff_guard import apply_resolved_edits
from article_rectifier.editing.minimal_edits import minimize_resolved_edit
from article_rectifier.editing.retry import merge_retry_results
from article_rectifier.editing.span_resolver import resolve_edit
from article_rectifier.models import (
    PatchResult,
    PatchStatus,
    SentenceSpan,
    SurgicalEdit,
)


def edit(old: str, new: str, identifier: str = "p") -> SurgicalEdit:
    """Build an unresolved replacement for an authorized test sentence."""
    return SurgicalEdit(
        patch_id=identifier,
        request_id="r",
        sentence_id="s",
        original_text=old,
        replacement_text=new,
        evidence_ids=("e",),
        error_type="numerical",
        explanation="A known numeric correction.",
    )


def sentence(text: str) -> SentenceSpan:
    """Create a span covering an exact fixture."""
    return SentenceSpan(
        sentence_id="s", text=text, start=0, end=len(text), paragraph_id="p"
    )


def test_three_number_sentence_becomes_three_independent_changes() -> None:
    """A verbose proposal must preserve all unchanged words and punctuation."""
    old = "Limits are 1,000cc and 1,200cc; tax is 12%."
    new = "Limits are 1,200cc and 1,500cc; tax is 18%."
    changes = minimize_resolved_edit(resolve_edit(old, sentence(old), edit(old, new)))
    assert len(changes) == 3
    assert {item.original_text for item in changes} == {"1,000cc", "1,200cc", "12"}
    assert apply_resolved_edits(old, changes) == new


def test_context_can_resolve_repetition_and_retry_keeps_resolved_offsets() -> None:
    """Accepted repeated values must not become ambiguous when another patch retries."""
    original = "Title: 55th meeting. Body: 55th meeting."
    first = minimize_resolved_edit(
        resolve_edit(
            original, sentence(original), edit("Title: 55th", "Title: 56th", "first")
        )
    )[0]
    second = minimize_resolved_edit(
        resolve_edit(
            original, sentence(original), edit("Body: 55th", "Body: 56th", "second")
        )
    )[0]
    accepted = [
        PatchResult(
            edit=item,
            status=PatchStatus.ACCEPTED,
            final_start=item.global_start,
            final_end=item.global_end,
        )
        for item in (first, second)
    ]
    text, results = merge_retry_results(original, [accepted[0]], [accepted[1]], 1, 1)
    assert text == "Title: 56th meeting. Body: 56th meeting."
    assert len(results) == 2
    assert results[0] == accepted[0]
