"""Authorize, resolve, audit, and execute surgical article edits."""

from __future__ import annotations

from dataclasses import dataclass

from article_rectifier.editing.diff_guard import (
    apply_resolved_edits,
    conflicting_patch_ids,
)
from article_rectifier.editing.span_resolver import resolve_edit
from article_rectifier.models.article import SentenceSpan
from article_rectifier.models.edits import SurgicalEdit
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import (
    PatchAuditRecord,
    PatchResult,
    PatchStatus,
    Verdict,
    VerificationResult,
    VerificationStatus,
)


@dataclass(frozen=True, slots=True)
class EditExecution:
    """Return reconstructed text, patch dispositions, and provenance records."""

    text: str
    results: tuple[PatchResult, ...]
    audit_records: tuple[PatchAuditRecord, ...]


def _authorization_error(
    edit: SurgicalEdit,
    verification: VerificationResult | None,
    evidence_by_id: dict[str, EvidenceSpan],
    source_text: str,
) -> str | None:
    """Check verdict, request, evidence identity, and source exactness.

    Args:
        edit: Proposed structured edit.
        verification: Judgment associated with the edit request.
        evidence_by_id: All retrieved exact source passages.
        source_text: Complete sanitized source article.

    Returns:
        Rejection reason, or None when authorization succeeds.
    """
    if verification is None or verification.request_id != edit.request_id:
        return "missing accepted verification"
    if verification.verdict is not Verdict.CONTRADICTED:
        return "verification is not a contradiction"
    if verification.status is not VerificationStatus.CORRECTION_AVAILABLE:
        return "verification has no safe correction"
    if not set(edit.evidence_ids) <= set(verification.evidence_ids):
        return "patch evidence was not accepted by verification"
    evidence = [evidence_by_id.get(identifier) for identifier in edit.evidence_ids]
    if any(item is None for item in evidence):
        return "patch references missing evidence"
    if any(not item.verify_against(source_text) for item in evidence if item):
        return "patch evidence does not match source text"
    return None


def _rejected(edit: SurgicalEdit, reason: str) -> PatchResult:
    """Create one explicit rejected patch result.

    Args:
        edit: Rejected proposed edit.
        reason: Deterministic rejection reason.

    Returns:
        Typed rejected patch result.
    """
    return PatchResult(edit=edit, status=PatchStatus.REJECTED, reason=reason)


def _audit(
    result: PatchResult,
    verification: VerificationResult | None,
) -> PatchAuditRecord:
    """Create human-inspectable provenance for one patch disposition.

    Args:
        result: Accepted or rejected patch result.
        verification: Associated factual judgment, when present.

    Returns:
        Complete patch audit record.
    """
    edit = result.edit
    verdict = verification.verdict if verification else Verdict.UNCERTAIN
    error_types = verification.error_types if verification else (edit.error_type,)
    return PatchAuditRecord(
        patch_id=edit.patch_id,
        sentence_id=edit.sentence_id,
        original_text=edit.original_text,
        replacement_text=edit.replacement_text,
        error_types=error_types,
        evidence_ids=edit.evidence_ids,
        verification_verdict=verdict,
        validation_status=result.status,
        reason=result.reason,
        start=result.final_start,
        end=result.final_end,
    )


def _resolve_authorized(
    original_text: str,
    proposed_edits: list[SurgicalEdit],
    sentences: dict[str, SentenceSpan],
    verifications: dict[str, VerificationResult],
    evidence_by_id: dict[str, EvidenceSpan],
    source_text: str,
) -> tuple[list[SurgicalEdit], list[PatchResult]]:
    """Authorize and resolve edits while retaining every rejection reason."""
    resolved: list[SurgicalEdit] = []
    rejected: list[PatchResult] = []
    for edit in proposed_edits:
        verification = verifications.get(edit.request_id)
        error = _authorization_error(edit, verification, evidence_by_id, source_text)
        sentence = sentences.get(edit.sentence_id)
        if error or sentence is None:
            rejected.append(_rejected(edit, error or "missing permitted sentence"))
            continue
        try:
            resolved.append(resolve_edit(original_text, sentence, edit))
        except ValueError as exception:
            rejected.append(_rejected(edit, str(exception)))
    return resolved, rejected


def execute_edits(
    original_text: str,
    proposed_edits: list[SurgicalEdit],
    sentences: list[SentenceSpan],
    verifications: list[VerificationResult],
    evidence: list[EvidenceSpan],
    source_text: str,
) -> EditExecution:
    """Apply all safe non-conflicting patches to the immutable original article.

    Args:
        original_text: Sanitized generated article before any patches.
        proposed_edits: Structured edits from the only proposal stage.
        sentences: Immutable editable sentence spans.
        verifications: Source-grounded request judgments.
        evidence: Exact source passages available for authorization.
        source_text: Complete sanitized source article.

    Returns:
        Reconstructed output plus accepted/rejected results and audit records.
    """
    sentence_map = {item.sentence_id: item for item in sentences}
    verification_map = {item.request_id: item for item in verifications}
    evidence_map = {item.evidence_id: item for item in evidence}
    resolved, rejected = _resolve_authorized(
        original_text,
        proposed_edits,
        sentence_map,
        verification_map,
        evidence_map,
        source_text,
    )
    conflicts = conflicting_patch_ids(resolved)
    accepted_edits = [edit for edit in resolved if edit.patch_id not in conflicts]
    rejected.extend(
        _rejected(edit, "patch overlaps another proposed edit")
        for edit in resolved
        if edit.patch_id in conflicts
    )
    output = apply_resolved_edits(original_text, accepted_edits)
    accepted = [
        PatchResult(
            edit=edit,
            status=PatchStatus.ACCEPTED,
            final_start=edit.global_start,
            final_end=edit.global_end,
        )
        for edit in accepted_edits
    ]
    results = sorted(accepted + rejected, key=lambda item: item.edit.patch_id)
    audits = [
        _audit(item, verification_map.get(item.edit.request_id)) for item in results
    ]
    return EditExecution(output, tuple(results), tuple(audits))
