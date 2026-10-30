"""Minimal patch proposal for independently verified factual errors."""

from __future__ import annotations

import json
from dataclasses import dataclass

from article_rectifier.llm.client import LLMCallRecord, LLMClient
from article_rectifier.llm.prompts import PATCH_PROMPT_VERSION, PATCH_SYSTEM_PROMPT
from article_rectifier.models.article import SentenceSpan
from article_rectifier.models.claims import FactCheckRequest
from article_rectifier.models.edits import EditOperation, PatchProposal, SurgicalEdit
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import (
    Verdict,
    VerificationResult,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class PatchProposalOutcome:
    """Return safe edit proposals with model-call provenance and errors."""

    proposal: PatchProposal
    call_record: LLMCallRecord | None
    errors: tuple[str, ...] = ()


def _is_actionable(verification: VerificationResult) -> bool:
    """Check whether a verification result permits patch proposal.

    Args:
        verification: Source-grounded factual judgment.

    Returns:
        True only for contradictions with a supported corrected fact.
    """
    return (
        verification.verdict is Verdict.CONTRADICTED
        and verification.status is VerificationStatus.CORRECTION_AVAILABLE
        and verification.proposed_factual_correction is not None
        and bool(verification.error_types)
    )


def _payload(
    sentence: SentenceSpan,
    request: FactCheckRequest,
    verification: VerificationResult,
    evidence: list[EvidenceSpan],
    retry_feedback: str | None = None,
    retry_count: int = 0,
) -> dict[str, object]:
    """Serialize only one sentence, verified issue, and cited evidence.

    Args:
        sentence: Permitted editable sentence.
        request: Original factual claim and question.
        verification: Actionable source-grounded contradiction.
        evidence: Exact passages cited by verification.
        retry_feedback: Targeted prior validation failure, when retrying.
        retry_count: Number of previous patch proposal attempts.

    Returns:
        JSON-compatible minimal patch task.
    """
    return {
        "permitted_sentence": sentence.model_dump(mode="json"),
        "request": request.model_dump(mode="json"),
        "verification": verification.model_dump(mode="json"),
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "retry_feedback": retry_feedback,
        "retry_count": retry_count,
        "constraints": {
            "smallest_exact_change": True,
            "style_changes_forbidden": True,
            "whole_sentence_deletion_last_resort": True,
            "global_offsets_must_be_null": True,
        },
        "response_schema": PatchProposal.model_json_schema(),
    }


def _edit_semantic_error(
    edit: SurgicalEdit,
    sentence: SentenceSpan,
    request: FactCheckRequest,
    verification: VerificationResult,
    evidence_ids: set[str],
) -> str | None:
    """Check one proposed edit against all permitted identities and spans.

    Args:
        edit: Schema-valid model patch.
        sentence: Only editable sentence.
        request: Verified request identity.
        verification: Grounded error categories and evidence.
        evidence_ids: Exact permitted evidence identifiers.

    Returns:
        Error code, or None when the proposal is semantically permitted.
    """
    if (
        edit.request_id != request.request_id
        or edit.sentence_id != sentence.sentence_id
    ):
        return "patch_identity_mismatch"
    if edit.global_start is not None or edit.global_end is not None:
        return "model_supplied_global_offsets"
    if not set(edit.evidence_ids) <= evidence_ids:
        return "patch_unknown_evidence"
    if edit.error_type not in verification.error_types:
        return "patch_error_type_mismatch"
    target = (
        edit.insertion_anchor
        if edit.operation is EditOperation.INSERT
        else edit.original_text
    )
    if target not in sentence.text:
        return "patch_target_outside_sentence"
    if (
        edit.operation is EditOperation.DELETE
        and edit.original_text == sentence.text
        and "unsupported" not in edit.explanation.casefold()
    ):
        return "sentence_deletion_not_justified"
    return None


def _validate_proposal(
    proposal: PatchProposal,
    sentence: SentenceSpan,
    request: FactCheckRequest,
    verification: VerificationResult,
    evidence: list[EvidenceSpan],
) -> tuple[list[SurgicalEdit], list[str]]:
    """Discard any proposal that exceeds the verified issue boundary.

    Args:
        proposal: Schema-valid model proposal.
        sentence: Only editable sentence.
        request: Verified request identity.
        verification: Grounded contradiction.
        evidence: Exact cited source passages.

    Returns:
        Safe edits and semantic rejection codes.
    """
    permitted_ids = set(verification.evidence_ids)
    available_ids = {item.evidence_id for item in evidence}
    evidence_ids = permitted_ids & available_ids
    accepted: list[SurgicalEdit] = []
    errors: list[str] = []
    for edit in proposal.edits:
        error = _edit_semantic_error(
            edit, sentence, request, verification, evidence_ids
        )
        if error:
            errors.append(f"{error}:{edit.patch_id}")
        else:
            accepted.append(edit)
    return accepted, errors


class PatchProposer:
    """Propose only small edits for actionable source-grounded contradictions."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the proposer with the centralized model adapter.

        Args:
            llm_client: Validated and observable LLM adapter.
        """
        self._llm = llm_client

    def propose(
        self,
        sentence: SentenceSpan,
        request: FactCheckRequest,
        verification: VerificationResult,
        evidence: list[EvidenceSpan],
        *,
        retry_feedback: str | None = None,
        retry_count: int = 0,
    ) -> PatchProposalOutcome:
        """Request and validate bounded edits for one verified error.

        Args:
            sentence: Only editable sentence.
            request: Original fact-check request.
            verification: Source-grounded factual judgment.
            evidence: Exact passages available to the proposer.
            retry_feedback: Targeted failure reason from the prior validation.
            retry_count: Number of previous proposal attempts.

        Returns:
            Safe patch proposals with provenance and rejection details.
        """
        if not _is_actionable(verification):
            return PatchProposalOutcome(PatchProposal(), None)
        messages = [
            {"role": "system", "content": PATCH_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": json.dumps(
                    _payload(
                        sentence,
                        request,
                        verification,
                        evidence,
                        retry_feedback,
                        retry_count,
                    ),
                    ensure_ascii=False,
                ),
            },
        ]
        response = self._llm.complete_structured(
            messages,
            PatchProposal,
            prompt_version=PATCH_PROMPT_VERSION,
        )
        if response.value is None:
            return PatchProposalOutcome(
                PatchProposal(),
                response.record,
                (f"patch_llm:{response.record.status}",),
            )
        accepted, errors = _validate_proposal(
            response.value, sentence, request, verification, evidence
        )
        return PatchProposalOutcome(
            PatchProposal(edits=tuple(accepted)),
            response.record,
            tuple(errors),
        )
