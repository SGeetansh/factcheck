"""Minimal patch proposal for independently verified factual errors."""

from __future__ import annotations

import json
import re
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
    raw_proposal: PatchProposal | None = None


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
        and (
            verification.proposed_factual_correction is not None
            or bool(verification.corrections)
        )
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

    def propose_group(
        self,
        sentence: SentenceSpan,
        requests: list[FactCheckRequest],
        verifications: list[VerificationResult],
        evidence: list[EvidenceSpan],
        *,
        retry_feedback: str | None = None,
        retry_count: int = 0,
    ) -> PatchProposalOutcome:
        """Coordinate overlapping verified issues in one sentence-level proposal.

        Args:
            sentence: The sole editable sentence.
            requests: Original requests for the actionable issues.
            verifications: Independently obtained actionable judgments.
            evidence: Exact passages available to those judgments.
            retry_feedback: Prior rejection reasons for this sentence.
            retry_count: Number of prior proposal attempts.

        Returns:
            Non-conflicting proposals, each authorized by its own verification.
        """
        direct = _fact_pair_proposal(sentence, verifications)
        if direct is not None and not retry_feedback:
            return PatchProposalOutcome(direct, None)
        if len(verifications) == 1:
            return self.propose(
                sentence,
                requests[0],
                verifications[0],
                evidence,
                retry_feedback=retry_feedback,
                retry_count=retry_count,
            )
        payload = {
            "permitted_sentence": sentence.model_dump(mode="json"),
            "verified_issues": [
                {
                    "request": request.model_dump(mode="json"),
                    "verification": verification.model_dump(mode="json"),
                }
                for request, verification in zip(requests, verifications, strict=True)
            ],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "retry_feedback": retry_feedback,
            "response_schema": PatchProposal.model_json_schema(),
        }
        response = self._llm.complete_structured(
            [
                {
                    "role": "system",
                    "content": PATCH_SYSTEM_PROMPT
                    + " Multiple checks can describe the SAME error. Correct it ONCE. "
                    "Return one coordinated set of non-overlapping edits for this "
                    "sentence. Each edit must use one supplied request_id and only "
                    "that request's verified evidence IDs and error types. Leave "
                    "global offsets null. Keep already-correct facts, currency "
                    "notation and style unchanged. Read the resulting sentence after "
                    "applying edits. Include an adjacent a/an in the target only "
                    "when the factual replacement requires that "
                    "agreement change. Do not leave a grammatically broken boundary.",
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            PatchProposal,
            prompt_version="patch-group-v2",
        )
        if response.value is None:
            return PatchProposalOutcome(
                PatchProposal(),
                response.record,
                (f"patch_llm:{response.record.status}",),
            )
        return _validate_group_response(
            response.value, response.record, sentence, requests, verifications, evidence
        )

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
            raw_proposal=response.value,
        )


def _validate_group_response(
    proposal: PatchProposal,
    record: LLMCallRecord,
    sentence: SentenceSpan,
    requests: list[FactCheckRequest],
    verifications: list[VerificationResult],
    evidence: list[EvidenceSpan],
) -> PatchProposalOutcome:
    """Validate each grouped edit against its own original evidence authorization.

    Args:
        proposal: Raw coordinated model proposal.
        record: Model-call provenance.
        sentence: Permitted immutable sentence.
        requests: Claims available to the proposer.
        verifications: Corresponding source judgments.
        evidence: Exact source passages.

    Returns:
        Individually authorized edits and explicit rejection codes.
    """
    by_id = {
        request.request_id: (request, verification)
        for request, verification in zip(requests, verifications, strict=True)
    }
    accepted: list[SurgicalEdit] = []
    errors: list[str] = []
    for edit in proposal.edits:
        pair = by_id.get(edit.request_id)
        if pair is None:
            errors.append(f"patch_identity_mismatch:{edit.patch_id}")
            continue
        edits, rejected = _validate_proposal(
            PatchProposal(edits=(edit,)), sentence, pair[0], pair[1], evidence
        )
        accepted.extend(edits)
        errors.extend(rejected)
    return PatchProposalOutcome(
        PatchProposal(edits=tuple(accepted)),
        record,
        tuple(errors),
        raw_proposal=proposal,
    )


def _fact_pair_proposal(
    sentence: SentenceSpan, verifications: list[VerificationResult]
) -> PatchProposal | None:
    """Build exact, independent replacements from verified value pairs.

    Args:
        sentence: Immutable editable sentence.
        verifications: Actionable results containing structured facts.

    Returns:
        Deduplicated surgical edits, or None when ambiguity needs model reasoning.
    """
    if not verifications or any(not item.corrections for item in verifications):
        return None
    edits: dict[tuple[str, str], SurgicalEdit] = {}
    targets: dict[str, str] = {}
    for verification in verifications:
        for fact in verification.corrections:
            if fact.original_value.endswith(
                (",", ";", ":")
            ) and not fact.correct_value.endswith(fact.original_value[-1]):
                fact = fact.model_copy(
                    update={
                        "correct_value": fact.correct_value + fact.original_value[-1]
                    }
                )
            if bool(re.search(r"\d", fact.original_value)) != bool(
                re.search(r"\d", fact.correct_value)
            ):
                return None
            if len(fact.correct_value) > len(fact.original_value) + 30:
                return None
            if sentence.text.count(fact.original_value) != 1:
                return None
            previous = targets.get(fact.original_value)
            if previous is not None and previous != fact.correct_value:
                return None
            if fact.original_value == fact.correct_value:
                continue
            targets[fact.original_value] = fact.correct_value
            key = (fact.original_value, fact.correct_value)
            edits.setdefault(
                key,
                SurgicalEdit(
                    patch_id=f"{verification.request_id}-fact-{len(edits) + 1}",
                    request_id=verification.request_id,
                    sentence_id=sentence.sentence_id,
                    operation=(
                        EditOperation.REPLACE
                        if fact.correct_value
                        else EditOperation.DELETE
                    ),
                    original_text=fact.original_value,
                    replacement_text=fact.correct_value,
                    evidence_ids=fact.evidence_ids,
                    error_type=fact.error_type,
                    explanation=verification.explanation,
                ),
            )
    spans = [
        (
            sentence.text.index(edit.original_text),
            sentence.text.index(edit.original_text) + len(edit.original_text),
        )
        for edit in edits.values()
    ]
    if any(
        a < d and c < b
        for index, (a, b) in enumerate(spans)
        for c, d in spans[index + 1 :]
    ):
        return None
    return PatchProposal(edits=tuple(edits.values()))
