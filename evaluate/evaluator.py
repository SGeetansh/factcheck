"""Top-level independent article evaluation interface."""

from __future__ import annotations

import re

from evaluate.annotation_metrics import correction_recall, detection_recall
from evaluate.factuality import unavailable_factuality
from evaluate.models import (
    AnnotationLabel,
    DetectedIssue,
    EvaluationReport,
    IntegrityResult,
    PatchObservation,
    UsageObservation,
)
from evaluate.patch_metrics import count_patches, edit_precision, issue_type_counts
from evaluate.preservation import (
    changed_character_ratio,
    levenshtein_distance,
    normalized_similarity,
)
from evaluate.reference_comparison import reference_similarity


def check_output_integrity(original: str, rectified: str) -> IntegrityResult:
    """Check deterministic article output constraints.

    Args:
        original: Sanitized generated article.
        rectified: Final rectified article.

    Returns:
        Integrity status and exact violation codes.
    """
    violations: list[str] = []
    if original and not rectified:
        violations.append("empty_output")
    if re.search(r"(?im)^\s*(?:\*\*)?error annotations:", rectified):
        violations.append("contains_error_annotations")
    if "```" in rectified:
        violations.append("contains_code_fence")
    stripped = rectified.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        violations.append("looks_like_json")
    return IntegrityResult(valid=not violations, violations=tuple(violations))


def evaluate_article(
    source: str,
    original: str,
    rectified: str,
    human_reference: str | None = None,
    *,
    annotations: list[AnnotationLabel] | None = None,
    detected_issues: list[DetectedIssue] | None = None,
    patches: list[PatchObservation] | None = None,
    unresolved_issues: int = 0,
    retrieval_failures: int = 0,
    usage: UsageObservation | None = None,
) -> EvaluationReport:
    """Evaluate one rectified article without live LLM calls.

    Args:
        source: Authoritative source text reserved for factuality evaluators.
        original: Sanitized generated article before edits.
        rectified: Final rectified article.
        human_reference: Optional human-corrected comparison text.
        annotations: Optional development-only error labels.
        detected_issues: Independently observable detected issues.
        patches: Proposed and applied patch observations.
        unresolved_issues: Observed unresolved issue count.
        retrieval_failures: Observed retrieval failure count.
        usage: Observed LLM usage and cost data.

    Returns:
        A typed report separating measured and unavailable metrics.
    """
    del source
    issues = detected_issues or []
    observed_patches = patches or []
    return EvaluationReport(
        preservation=normalized_similarity(original, rectified, "preservation"),
        character_edit_distance=levenshtein_distance(original, rectified),
        changed_character_ratio=changed_character_ratio(original, rectified),
        human_reference_similarity=reference_similarity(rectified, human_reference),
        annotation_detection_recall=detection_recall(annotations, issues),
        correction_recall=correction_recall(annotations, rectified),
        proposed_edit_precision=edit_precision(observed_patches, applied_only=False),
        applied_edit_precision=edit_precision(observed_patches, applied_only=True),
        factuality=unavailable_factuality(),
        patch_counts=count_patches(
            observed_patches,
            unresolved_issues=unresolved_issues,
            retrieval_failures=retrieval_failures,
        ),
        errors_detected_by_type=issue_type_counts(issues),
        integrity=check_output_integrity(original, rectified),
        usage=usage or UsageObservation(),
    )
