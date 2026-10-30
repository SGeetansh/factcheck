"""Source-grounded factual verification with conservative safety guards."""

from __future__ import annotations

import json
from dataclasses import dataclass

from article_rectifier.llm.client import LLMCallRecord, LLMClient
from article_rectifier.llm.prompts import (
    VERIFIER_PROMPT_VERSION,
    VERIFIER_SYSTEM_PROMPT,
)
from article_rectifier.models.claims import FactCheckRequest
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import (
    Verdict,
    VerificationResult,
    VerificationStatus,
)
from article_rectifier.retrieval.source_index import tokenize


@dataclass(frozen=True, slots=True)
class VerificationOutcome:
    """Return one safe verdict with model-call provenance and errors."""

    result: VerificationResult
    call_record: LLMCallRecord | None
    errors: tuple[str, ...] = ()
    raw_result: VerificationResult | None = None


def _unresolved_result(
    request: FactCheckRequest,
    verdict: Verdict,
    explanation: str,
    evidence_ids: tuple[str, ...] = (),
) -> VerificationResult:
    """Create a safe non-actionable verification result.

    Args:
        request: Original fact-check request.
        verdict: Unsupported or uncertain verdict.
        explanation: Short reason for leaving the claim unchanged.
        evidence_ids: Relevant exact passages, when any.

    Returns:
        A typed unresolved verification result.
    """
    return VerificationResult(
        request_id=request.request_id,
        verdict=verdict,
        evidence_ids=evidence_ids,
        explanation=explanation,
        status=VerificationStatus.UNRESOLVED,
    )


def _evidence_is_relevant(
    request: FactCheckRequest,
    evidence: list[EvidenceSpan],
) -> bool:
    """Reject passages with no lexical link to the requested proposition.

    Args:
        request: Claim and stable search terms.
        evidence: Retrieved source passages.

    Returns:
        True when at least one informative request token occurs in evidence.
    """
    request_tokens = set(tokenize(f"{request.claim_text} {request.search_query}"))
    evidence_tokens = set(tokenize(" ".join(item.text for item in evidence)))
    return bool(request_tokens & evidence_tokens)


def _correction_has_evidence_anchor(
    correction: str, evidence: list[EvidenceSpan]
) -> bool:
    """Check that a suggested fact shares content with cited passages.

    This is only a coarse gate. The patch validator must independently check
    the exact replacement value before accepting an edit.

    Args:
        correction: Model-proposed factual correction, possibly a full sentence.
        evidence: Exact evidence passages cited for that correction.

    Returns:
        True when at least one non-stopword token overlaps the evidence.
    """
    correction_tokens = set(tokenize(correction))
    evidence_tokens = set(tokenize(" ".join(item.text for item in evidence)))
    return bool(correction_tokens & evidence_tokens)


def _payload(
    request: FactCheckRequest,
    evidence: list[EvidenceSpan],
) -> dict[str, object]:
    """Serialize one bounded verification task.

    Args:
        request: Exact claim and verification question.
        evidence: Exact retrieved source passages.

    Returns:
        JSON-compatible prompt payload.
    """
    return {
        "request": request.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "response_schema": VerificationResult.model_json_schema(),
    }


def _semantic_error(
    result: VerificationResult,
    request: FactCheckRequest,
    evidence: list[EvidenceSpan],
) -> str | None:
    """Validate request identity, evidence citations, and corrected facts.

    Args:
        result: Schema-valid model verdict.
        request: Original request.
        evidence: Permitted source passages.

    Returns:
        Error code for an unsafe result, or None when safe.
    """
    if result.request_id != request.request_id:
        return "request_id_mismatch"
    evidence_by_id = {item.evidence_id: item for item in evidence}
    if any(identifier not in evidence_by_id for identifier in result.evidence_ids):
        return "unknown_evidence_id"
    correction = result.proposed_factual_correction
    if correction:
        cited = [evidence_by_id[identifier] for identifier in result.evidence_ids]
        if not _correction_has_evidence_anchor(correction, cited):
            return "correction_not_in_evidence"
    return None


class EvidenceVerifier:
    """Judge source support without permitting edits or external knowledge."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the verifier with the centralized model adapter.

        Args:
            llm_client: Validated and observable LLM adapter.
        """
        self._llm = llm_client

    def verify(
        self,
        request: FactCheckRequest,
        evidence: list[EvidenceSpan],
        source_text: str,
    ) -> VerificationOutcome:
        """Verify one claim against exact supplied source evidence.

        Args:
            request: Fact-check request from the planner.
            evidence: Deterministically retrieved source passages.
            source_text: Complete sanitized source used for offset checks.

        Returns:
            Safe typed verdict with provenance and semantic errors.
        """
        valid = [item for item in evidence if item.verify_against(source_text)]
        if len(valid) != len(evidence):
            result = _unresolved_result(
                request, Verdict.UNCERTAIN, "Evidence offsets failed source validation."
            )
            return VerificationOutcome(result, None, ("invalid_evidence_span",))
        if not valid or not _evidence_is_relevant(request, valid):
            result = _unresolved_result(
                request,
                Verdict.UNSUPPORTED,
                "Retrieved evidence does not answer the question.",
            )
            return VerificationOutcome(result, None)
        messages = [
            {"role": "system", "content": VERIFIER_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(_payload(request, valid), ensure_ascii=False),
            },
        ]
        response = self._llm.complete_structured(
            messages,
            VerificationResult,
            prompt_version=VERIFIER_PROMPT_VERSION,
        )
        if response.value is None:
            result = _unresolved_result(
                request,
                Verdict.UNCERTAIN,
                "Verification model did not return a valid result.",
            )
            return VerificationOutcome(
                result, response.record, (f"verifier_llm:{response.record.status}",)
            )
        error = _semantic_error(response.value, request, valid)
        if error:
            result = _unresolved_result(
                request,
                Verdict.UNCERTAIN,
                "Verification response failed source-grounding checks.",
                tuple(item.evidence_id for item in valid),
            )
            return VerificationOutcome(
                result, response.record, (error,), raw_result=response.value
            )
        return VerificationOutcome(
            response.value, response.record, raw_result=response.value
        )
