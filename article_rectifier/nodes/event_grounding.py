"""Check exact correction witnesses and explicit event-year compatibility."""

import re

from article_rectifier.models.claims import FactCheckRequest
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import VerificationResult


def event_grounding_error(
    result: VerificationResult, request: FactCheckRequest, evidence: list[EvidenceSpan]
) -> str | None:
    """Reject unquoted corrections and evidence from a different explicit year.

    Args:
        result: Proposed structured factual judgment.
        request: Original claim with its event context.
        evidence: Exact source passages available to the verifier.

    Returns:
        A rejection code or None. Passing is not an entailment proof.
    """
    passages = {item.evidence_id: item.text for item in evidence}
    if result.status.value == "correction_available" and not result.corrections:
        return "correction_missing_structured_witness"
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", request.claim_text))
    corrected_years = {
        year
        for fact in result.corrections
        for year in re.findall(r"\b(?:19|20)\d{2}\b", fact.original_value)
    }
    retained_years = years - corrected_years
    for fact in result.corrections:
        negative = r"\b(?:not|never|no)\b|n't\b"
        if (
            re.search(negative, fact.correct_value, re.I)
            and not re.search(negative, fact.original_value, re.I)
            and not re.search(negative, fact.support_quote, re.I)
        ):
            return "correction_negative_not_established"
        if not fact.support_quote.strip() or not any(
            fact.support_quote in passages.get(identifier, "")
            for identifier in fact.evidence_ids
        ):
            return "correction_missing_exact_support_quote"
        witness_years = set(re.findall(r"\b(?:19|20)\d{2}\b", fact.support_quote))
        if retained_years and not retained_years <= witness_years:
            return "correction_event_year_not_established"
    return None
