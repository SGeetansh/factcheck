"""Independent structural, factual, and preservation patch validation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from article_rectifier.editing.diff_guard import apply_resolved_edits
from article_rectifier.models.article import SentenceSpan
from article_rectifier.models.edits import EditOperation, SurgicalEdit
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import PatchResult, PatchStatus
from article_rectifier.retrieval.source_index import tokenize


@dataclass(frozen=True, slots=True)
class ValidationExecution:
    """Return safe output and final validation disposition for every patch."""

    text: str
    results: tuple[PatchResult, ...]
    warnings: tuple[str, ...] = ()


def _accepted_edits(results: list[PatchResult]) -> list[SurgicalEdit]:
    """Select edits provisionally accepted by deterministic execution.

    Args:
        results: Patch execution dispositions.

    Returns:
        Resolved edits with accepted status.
    """
    return [item.edit for item in results if item.status is PatchStatus.ACCEPTED]


def validate_reconstruction(
    original_text: str,
    candidate_text: str,
    results: list[PatchResult],
) -> str | None:
    """Check that no text outside accepted patches changed.

    Args:
        original_text: Immutable sanitized generated article.
        candidate_text: Provisional edited output.
        results: Patch execution dispositions.

    Returns:
        Error reason, or None for exact reconstruction.
    """
    try:
        expected = apply_resolved_edits(original_text, _accepted_edits(results))
    except ValueError as error:
        return str(error)
    if expected != candidate_text:
        return "candidate output differs from exact patch reconstruction"
    if original_text and not candidate_text:
        return "non-empty article became empty"
    if "```" in candidate_text:
        return "candidate output contains a code fence"
    if re.search(r"(?im)^\s*(?:\*\*)?error annotations:", candidate_text):
        return "candidate output contains error annotations"
    return None


def _cited_evidence(
    edit: SurgicalEdit,
    evidence_by_id: dict[str, EvidenceSpan],
    source_text: str,
) -> tuple[list[EvidenceSpan], str | None]:
    """Resolve and revalidate all evidence cited by one edit.

    Args:
        edit: Resolved edit under validation.
        evidence_by_id: Available evidence by stable identity.
        source_text: Complete sanitized source article.

    Returns:
        Cited evidence and an optional failure reason.
    """
    cited = [evidence_by_id.get(identifier) for identifier in edit.evidence_ids]
    if any(item is None for item in cited):
        return [], "missing cited evidence"
    exact = [item for item in cited if item is not None]
    if any(not item.verify_against(source_text) for item in exact):
        return [], "cited evidence does not match source text"
    return exact, None


def _replacement_is_grounded(edit: SurgicalEdit, evidence: list[EvidenceSpan]) -> bool:
    """Check exact correction values against source evidence.

    Args:
        edit: Resolved replacement or insertion.
        evidence: Revalidated cited passages.

    Returns:
        True when replacement text is an exact phrase or token subset.
    """
    combined = " ".join(item.text for item in evidence).casefold()
    replacement = edit.replacement_text.strip().casefold()
    if replacement and replacement in combined:
        return True
    replacement_tokens = set(tokenize(replacement))
    evidence_tokens = set(tokenize(combined))
    if not replacement_tokens or not replacement_tokens <= evidence_tokens:
        return False
    numbers = set(re.findall(r"\d+(?:[.,]\d+)*", replacement))
    evidence_numbers = set(re.findall(r"\d+(?:[.,]\d+)*", combined))
    return numbers <= evidence_numbers


def _corrected_sentence(
    original_text: str,
    sentence: SentenceSpan,
    edit: SurgicalEdit,
) -> str:
    """Apply one edit to its sentence for targeted deletion validation.

    Args:
        original_text: Immutable article text.
        sentence: Permitted sentence span.
        edit: Resolved edit.

    Returns:
        Corrected sentence text.
    """
    if edit.global_start is None or edit.global_end is None:
        return sentence.text
    local_start = edit.global_start - sentence.start
    local_end = edit.global_end - sentence.start
    return (
        sentence.text[:local_start] + edit.replacement_text + sentence.text[local_end:]
    )


def _deletion_is_grounded(
    original_text: str,
    sentence: SentenceSpan,
    edit: SurgicalEdit,
    evidence: list[EvidenceSpan],
) -> bool:
    """Accept deletion only when the corrected sentence occurs verbatim in evidence.

    Args:
        original_text: Immutable article text.
        sentence: Edited sentence span.
        edit: Resolved deletion.
        evidence: Revalidated cited passages.

    Returns:
        True only for exact source support of the post-deletion sentence.
    """
    corrected = _corrected_sentence(original_text, sentence, edit).strip().casefold()
    return any(corrected and corrected in item.text.casefold() for item in evidence)


def _factual_error(
    original_text: str,
    edit: SurgicalEdit,
    sentence: SentenceSpan | None,
    evidence: list[EvidenceSpan],
) -> str | None:
    """Run conservative deterministic corrected-fact checks.

    Args:
        original_text: Immutable article text.
        edit: Resolved edit.
        sentence: Permitted immutable sentence.
        evidence: Revalidated cited passages.

    Returns:
        Failure reason, or None when deterministic grounding succeeds.
    """
    if sentence is None:
        return "missing sentence metadata"
    if edit.operation is EditOperation.DELETE:
        if not _deletion_is_grounded(original_text, sentence, edit, evidence):
            return "deletion lacks exact corrected-sentence support"
        return None
    if not _replacement_is_grounded(edit, evidence):
        return "replacement value is absent from cited evidence"
    return None


def _failed_validation_result(
    result: PatchResult,
    reason: str,
    retry_count: int,
    max_patch_retries: int,
) -> PatchResult:
    """Choose bounded retry or unresolved status for one factual failure.

    Args:
        result: Provisionally accepted patch.
        reason: Deterministic validation failure.
        retry_count: Prior proposal retry count.
        max_patch_retries: Configured maximum targeted retries.

    Returns:
        Retryable or unresolved patch result.
    """
    status = (
        PatchStatus.RETRYABLE
        if retry_count < max_patch_retries
        else PatchStatus.UNRESOLVED
    )
    return PatchResult(edit=result.edit, status=status, reason=reason)


def validate_patches(
    original_text: str,
    candidate_text: str,
    patch_results: list[PatchResult],
    sentences: list[SentenceSpan],
    evidence: list[EvidenceSpan],
    source_text: str,
    *,
    retry_count: int = 0,
    max_patch_retries: int = 1,
    soft_patch_character_limit: int = 120,
) -> ValidationExecution:
    """Validate patches and rebuild output from only final accepted edits.

    Args:
        original_text: Immutable sanitized generated article.
        candidate_text: Provisional editor output.
        patch_results: Editor patch dispositions.
        sentences: Immutable sentence metadata.
        evidence: Exact source evidence.
        source_text: Complete sanitized source article.
        retry_count: Prior targeted proposal retries.
        max_patch_retries: Maximum permitted retries.
        soft_patch_character_limit: Warning threshold for large edits.

    Returns:
        Safe reconstructed output and final patch statuses.
    """
    structural_error = validate_reconstruction(
        original_text, candidate_text, patch_results
    )
    if structural_error:
        failed = [
            (
                PatchResult(
                    edit=item.edit, status=PatchStatus.REJECTED, reason=structural_error
                )
                if item.status is PatchStatus.ACCEPTED
                else item
            )
            for item in patch_results
        ]
        return ValidationExecution(original_text, tuple(failed))
    return _validate_individual_patches(
        original_text,
        patch_results,
        sentences,
        evidence,
        source_text,
        retry_count,
        max_patch_retries,
        soft_patch_character_limit,
    )


def _validate_individual_patches(
    original_text: str,
    patch_results: list[PatchResult],
    sentences: list[SentenceSpan],
    evidence: list[EvidenceSpan],
    source_text: str,
    retry_count: int,
    max_patch_retries: int,
    soft_patch_character_limit: int,
) -> ValidationExecution:
    """Validate evidence and corrected facts for each accepted patch."""
    sentence_map = {item.sentence_id: item for item in sentences}
    evidence_map = {item.evidence_id: item for item in evidence}
    final_results: list[PatchResult] = []
    warnings: list[str] = []
    for result in patch_results:
        if result.status is not PatchStatus.ACCEPTED:
            final_results.append(result)
            continue
        cited, error = _cited_evidence(result.edit, evidence_map, source_text)
        if error is None:
            error = _factual_error(
                original_text,
                result.edit,
                sentence_map.get(result.edit.sentence_id),
                cited,
            )
        if error:
            final_results.append(
                _failed_validation_result(result, error, retry_count, max_patch_retries)
            )
            continue
        if (
            max(len(result.edit.original_text), len(result.edit.replacement_text))
            > soft_patch_character_limit
        ):
            warnings.append(f"large_patch:{result.edit.patch_id}")
        final_results.append(result)
    accepted = _accepted_edits(final_results)
    output = apply_resolved_edits(original_text, accepted)
    return ValidationExecution(output, tuple(final_results), tuple(warnings))
