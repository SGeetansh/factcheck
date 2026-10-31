"""Do not confuse request identity with complete correction coverage."""

from typing import cast

from article_rectifier.editing.resolution import unresolved_results
from article_rectifier.models.article import ArticleBatch
from article_rectifier.models.claims import FactCheckRequest
from article_rectifier.models.edits import SurgicalEdit
from article_rectifier.models.results import (
    FactCorrection,
    PatchResult,
    VerificationResult,
)
from article_rectifier.models.state import BatchState
from article_rectifier.preprocessing.sentence_splitter import split_sentences


def fixture_state(patch_request: str = "duplicate") -> BatchState:
    """Return a batch where one of two independently wrong facts was patched."""
    text = "Speed is 50,000 rpm and power is 1,500 kW."
    sentence = split_sentences(text)[0]
    facts = tuple(
        FactCorrection(
            original_value=old,
            correct_value=new,
            evidence_ids=("e",),
            error_type="numerical",
            support_quote="Fixture source",
        )
        for old, new in (("50,000 rpm", "25,000 rpm"), ("1,500 kW", "150 kW"))
    )
    request = FactCheckRequest(
        request_id="r",
        sentence_id=sentence.sentence_id,
        claim_text=text,
        verification_question="Which specifications?",
        search_query="speed power",
        sentence_start=0,
        sentence_end=len(text),
    )
    result = VerificationResult(
        request_id="r",
        verdict="contradicted",
        evidence_ids=("e",),
        explanation="Source differs",
        error_types=("numerical",),
        status="correction_available",
        corrections=facts,
    )
    start = text.index("50,000")
    edit = SurgicalEdit(
        patch_id="p",
        request_id=patch_request,
        sentence_id=sentence.sentence_id,
        original_text="50,000",
        replacement_text="25,000",
        evidence_ids=("e",),
        error_type="numerical",
        explanation="Source value",
        global_start=start,
        global_end=start + 6,
    )
    return cast(
        BatchState,
        {
            "batch": ArticleBatch(batch_id="b", sentences=(sentence,)),
            "fact_requests": [request],
            "verification_results": [result],
            "patch_results": [
                PatchResult(
                    edit=edit, status="accepted", final_start=start, final_end=start + 6
                )
            ],
        },
    )


def test_duplicate_request_is_resolved_by_another_accepted_patch() -> None:
    """A different request ID can implement the complete verified substitution."""
    state = fixture_state()
    result = state["verification_results"][0]
    state["verification_results"] = [
        result.model_copy(update={"corrections": result.corrections[:1]})
    ]
    assert unresolved_results(state) == []


def test_partial_correction_stays_unresolved_even_with_matching_request_id() -> None:
    """An accepted speed patch cannot hide the still-wrong power value."""
    state = fixture_state("r")
    assert unresolved_results(state) == state["verification_results"]


def test_rejected_patch_does_not_resolve_a_claim() -> None:
    """A proposed correct value counts only when the final patch was accepted."""
    state = fixture_state()
    state["patch_results"] = [
        state["patch_results"][0].model_copy(
            update={"status": "rejected", "reason": "Unsafe"}
        )
    ]
    assert unresolved_results(state) == state["verification_results"]


def test_matching_value_in_another_sentence_does_not_resolve_claim() -> None:
    """Correction accounting is tied to exact original sentence and fact offsets."""
    state = fixture_state()
    patch = state["patch_results"][0]
    state["patch_results"] = [
        patch.model_copy(
            update={"edit": patch.edit.model_copy(update={"sentence_id": "other"})}
        )
    ]
    assert unresolved_results(state) == state["verification_results"]
