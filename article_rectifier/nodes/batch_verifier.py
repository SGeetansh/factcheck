"""Verify a three-sentence batch in one tracked, structured model request."""

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from article_rectifier.llm.client import LLMCallRecord, LLMClient
from article_rectifier.llm.prompts import VERIFIER_SYSTEM_PROMPT
from article_rectifier.models.claims import FactCheckRequest
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import Verdict, VerificationResult
from article_rectifier.nodes.event_grounding import event_grounding_error
from article_rectifier.nodes.evidence_verifier import (
    _semantic_error,
    _unresolved_result,
)


class VerificationBatch(BaseModel):
    """Require explicit judgments; a missing result never means supported."""

    model_config = ConfigDict(extra="forbid")
    results: tuple[VerificationResult, ...]


@dataclass(frozen=True)
class BatchVerificationOutcome:
    """Retain validated judgments, errors, and one shared usage record."""

    results: tuple[VerificationResult, ...]
    record: LLMCallRecord | None
    errors: tuple[str, ...]


def verify_batch(
    client: LLMClient,
    requests: list[FactCheckRequest],
    evidence: dict[str, list[EvidenceSpan]],
    source_text: str,
    *,
    recovery: bool = False,
    article_context: str = "",
) -> BatchVerificationOutcome:
    """Verify every claim using shared evidence without repeated model calls.

    Args:
        client: Tracked model adapter.
        requests: Deduplicated questions for at most three sentences.
        evidence: Exact passages authorized separately for each request.
        source_text: Immutable source used to recheck offsets.
        recovery: Ask what the source establishes for previously unresolved facts.
        article_context: Complete editable sentences and adjacent read-only context.

    Returns:
        One judgment per requested identity and an explicit call record.
    """
    if not requests:
        return BatchVerificationOutcome((), None, ())
    unique = {item.evidence_id: item for spans in evidence.values() for item in spans}
    if any(not item.verify_against(source_text) for item in unique.values()):
        raise ValueError("Batch verification received invalid source evidence")
    payload = {
        "article_context": article_context,
        "requests": [request.model_dump(mode="json") for request in requests],
        "retrieved_for": {
            key: [item.evidence_id for item in spans] for key, spans in evidence.items()
        },
        "evidence": [
            item.model_dump(mode="json", exclude={"retrieval_score"})
            for item in unique.values()
        ],
        "response_schema": VerificationBatch.model_json_schema(),
    }
    instruction = VERIFIER_SYSTEM_PROMPT + (
        " Align each article claim with the SAME event in the source, using the "
        "complete article_context to resolve purpose, chronology and relationships. "
        "The task is to correct factual substitutions, not merely search for the "
        "literal incorrect name. Verify ALL facts, including numbers and comparisons. "
        "Return one result for every request_id. For contradictions, populate "
        "corrections with original_value copied EXACTLY from the claim, correct_value, "
        "evidence_ids, and error_type. Return facts, not rewritten sentences. Split "
        "independent numbers, "
        "names, dates and reversed relationships into independent minimal corrections. "
        "Keep explanations under 160 characters. Leave proposed_factual_correction "
        "null "
        "when corrections are present. If several requests identify the same error, "
        "return the same minimal value pair; Python will deduplicate it. "
        "Cite any evidence ID in the shared evidence list; never invent an ID. "
        "Every correction MUST include support_quote: an exact contiguous source "
        "quotation establishing the corrected fact for the SAME event. If the claim "
        "names a year that is not being corrected, the quote must establish that year. "
        "Do not transfer a venue, participant, or result between different editions "
        "of a tournament or different visits. Return uncertain when event identity "
        "cannot be established."
    )
    if recovery:
        instruction += (
            " This is a bounded recovery pass with broader source context. Answer "
            "what the source DOES establish for the specific event or relationship. "
            "An incorrect entity may be absent because a DIFFERENT venue, event, "
            "person or value is correct. Resolve the matching event using its purpose "
            "and surrounding context. Return that minimal correction when established. "
            "Do not confuse different events, and do not infer a contradiction from "
            "mere absence. Leave genuinely uncovered or ambiguous facts unresolved."
        )
    response = client.complete_structured(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        VerificationBatch,
        prompt_version=(
            "batch-verifier-recovery-v3" if recovery else "batch-verifier-v4"
        ),
    )
    return _validate_batch(requests, evidence, response.value, response.record)


def _validate_batch(
    requests: list[FactCheckRequest],
    evidence: dict[str, list[EvidenceSpan]],
    response: VerificationBatch | None,
    record: LLMCallRecord,
) -> BatchVerificationOutcome:
    """Reject missing, duplicate or unauthorized judgments without guessing values."""
    raw = response.results if response else ()
    shared = list(
        {
            item.evidence_id: item for spans in evidence.values() for item in spans
        }.values()
    )
    results = []
    errors = []
    known_ids = {request.request_id for request in requests}
    errors.extend(
        f"unknown_verification:{item.request_id}"
        for item in raw
        if item.request_id not in known_ids
    )
    for request in requests:
        matches = [item for item in raw if item.request_id == request.request_id]
        error = (
            _semantic_error(matches[0], request, shared)
            if len(matches) == 1
            else "missing_or_duplicate_verification"
        )
        if error is None:
            error = event_grounding_error(matches[0], request, shared)
        if error:
            errors.append(f"{error}:{request.request_id}")
            results.append(_unresolved_result(request, Verdict.UNCERTAIN, error))
        else:
            results.append(matches[0])
    return BatchVerificationOutcome(tuple(results), record, tuple(errors))
