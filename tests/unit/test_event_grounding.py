"""Prevent venue corrections borrowed from another tournament edition."""

from article_rectifier.models.claims import FactCheckRequest
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import VerificationResult
from article_rectifier.nodes.event_grounding import event_grounding_error


def test_different_year_cannot_authorize_a_venue_change() -> None:
    """An upcoming UAE event does not establish the venue of a 2023 event."""
    source = "The 2025 Asia Cup will take place in the UAE."
    request = FactCheckRequest(
        request_id="r",
        sentence_id="s",
        claim_text="They visited Pakistan for the 2023 Asia Cup.",
        verification_question="Where was the 2023 event?",
        search_query="Asia Cup",
        sentence_start=0,
        sentence_end=50,
    )
    evidence = EvidenceSpan(
        evidence_id="e",
        source_article_id="a",
        text=source,
        start=0,
        end=len(source),
        retrieval_score=1,
        retrieval_strategy="bm25",
    )
    result = VerificationResult.model_validate(
        {
            "request_id": "r",
            "verdict": "contradicted",
            "evidence_ids": ["e"],
            "explanation": "Venue correction",
            "error_types": ["geographic"],
            "status": "correction_available",
            "corrections": [
                {
                    "original_value": "Pakistan",
                    "correct_value": "UAE",
                    "evidence_ids": ["e"],
                    "error_type": "geographic",
                    "support_quote": source,
                }
            ],
        }
    )
    assert (
        event_grounding_error(result, request, [evidence])
        == "correction_event_year_not_established"
    )
    matching = source.replace("2025", "2023")
    evidence = evidence.model_copy(update={"text": matching})
    result = result.model_copy(
        update={
            "corrections": (
                result.corrections[0].model_copy(update={"support_quote": matching}),
            )
        }
    )
    assert event_grounding_error(result, request, [evidence]) is None


def test_invented_quote_is_rejected() -> None:
    """Reject a supplied quote that does not occur in the source."""
    from article_rectifier.models.results import FactCorrection

    request = FactCheckRequest(
        request_id="r",
        sentence_id="s",
        claim_text="Count 20.",
        verification_question="What count?",
        search_query="count",
        sentence_start=0,
        sentence_end=9,
    )
    result = VerificationResult(
        request_id="r",
        verdict="contradicted",
        evidence_ids=("e",),
        explanation="Count",
        error_types=("numerical",),
        status="correction_available",
        corrections=(
            FactCorrection(
                original_value="20",
                correct_value="10",
                evidence_ids=("e",),
                error_type="numerical",
                support_quote="Count 10.",
            ),
        ),
    )
    assert (
        event_grounding_error(result, request, [])
        == "correction_missing_exact_support_quote"
    )


def test_absence_does_not_establish_nonattendance() -> None:
    """A positive donation fact cannot authorize a negative attendance claim."""
    request = FactCheckRequest(
        request_id="r",
        sentence_id="s",
        claim_text="attendance at Event A",
        verification_question="Which event?",
        search_query="event",
        sentence_start=0,
        sentence_end=21,
    )
    source = "He donated to Event A."
    evidence = EvidenceSpan(
        evidence_id="e",
        source_article_id="a",
        text=source,
        start=0,
        end=len(source),
        retrieval_score=1,
        retrieval_strategy="bm25",
    )
    result = VerificationResult.model_validate(
        {
            "request_id": "r",
            "verdict": "contradicted",
            "evidence_ids": ["e"],
            "explanation": "Attendance",
            "error_types": ["other"],
            "status": "correction_available",
            "corrections": [
                {
                    "original_value": "attendance at",
                    "correct_value": "not attending",
                    "evidence_ids": ["e"],
                    "error_type": "other",
                    "support_quote": source,
                }
            ],
        }
    )
    assert (
        event_grounding_error(result, request, [evidence])
        == "correction_negative_not_established"
    )
    empty = result.model_copy(update={"corrections": ()})
    assert (
        event_grounding_error(empty, request, [evidence])
        == "correction_missing_structured_witness"
    )
