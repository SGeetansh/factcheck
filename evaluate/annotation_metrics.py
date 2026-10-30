"""Development-only annotation matching metrics."""

from __future__ import annotations

import re
import unicodedata

from evaluate.models import AnnotationLabel, DetectedIssue, ScoreResult
from evaluate.preservation import unavailable_score


def normalize_match_text(text: str) -> str:
    """Normalize text for conservative annotation-to-issue matching.

    Args:
        text: Text from an annotation or detected issue.

    Returns:
        Case-folded alphanumeric tokens separated by one space.
    """
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(re.findall(r"[\w₹$%]+", normalized, flags=re.UNICODE))


def issue_matches_annotation(issue: DetectedIssue, label: AnnotationLabel) -> bool:
    """Match an issue to an annotation using its factual phrase.

    Args:
        issue: Detected factual issue.
        label: Development annotation.

    Returns:
        True when one normalized factual phrase contains the other.
    """
    claim = normalize_match_text(issue.claim_text)
    error = normalize_match_text(label.error_text)
    return bool(claim and error and (claim in error or error in claim))


def matched_annotation_ids(
    annotations: list[AnnotationLabel],
    issues: list[DetectedIssue],
) -> set[str]:
    """Find applicable annotation IDs matched by distinct detected issues.

    Args:
        annotations: Provisional development labels.
        issues: Issues produced by the rectifier.

    Returns:
        IDs of matched applicable annotations.
    """
    remaining = set(range(len(issues)))
    matched: set[str] = set()
    for label in annotations:
        if not label.applicable:
            continue
        for index in sorted(remaining):
            if issue_matches_annotation(issues[index], label):
                matched.add(label.annotation_id)
                remaining.remove(index)
                break
    return matched


def detection_recall(
    annotations: list[AnnotationLabel] | None,
    issues: list[DetectedIssue],
) -> ScoreResult:
    """Calculate provisional annotation detection recall.

    Args:
        annotations: Optional development annotations.
        issues: Detected factual issues.

    Returns:
        Recall over applicable annotations, or unavailable when labels are absent.
    """
    if annotations is None:
        return unavailable_score(
            "annotation_detection_recall",
            "No development annotations were supplied.",
        )
    applicable = [label for label in annotations if label.applicable]
    matched = matched_annotation_ids(applicable, issues)
    value = 1.0 if not applicable else len(matched) / len(applicable)
    return ScoreResult(
        name="annotation_detection_recall",
        value=value,
        numerator=len(matched),
        denominator=len(applicable),
        note="Development annotations are provisional and may be incomplete.",
    )


def correction_recall(
    annotations: list[AnnotationLabel] | None,
    rectified: str,
) -> ScoreResult:
    """Calculate conservative correction recall from exact validated labels.

    Args:
        annotations: Optional development annotations.
        rectified: Final rectified article.

    Returns:
        Recall based on absent error phrases and present correction phrases.
    """
    if annotations is None:
        return unavailable_score(
            "correction_recall",
            "No validated correction labels were supplied.",
        )
    applicable = [label for label in annotations if label.applicable]
    normalized = normalize_match_text(rectified)
    corrected = sum(
        normalize_match_text(label.error_text) not in normalized
        and normalize_match_text(label.correction_text) in normalized
        for label in applicable
    )
    value = 1.0 if not applicable else corrected / len(applicable)
    return ScoreResult(
        name="correction_recall",
        value=value,
        numerator=corrected,
        denominator=len(applicable),
        note=(
            "Exact label matching requires human validation for paraphrased "
            "corrections."
        ),
    )
