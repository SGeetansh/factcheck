"""Tests for independent article evaluation."""

import pytest

from evaluate import evaluate_article
from evaluate.models import (
    AnnotationLabel,
    Availability,
    DetectedIssue,
    PatchObservation,
)


def test_evaluate_article_without_reference_or_llm() -> None:
    """Produce deterministic metrics and explicit unavailable judgments.

    Returns:
        None.
    """
    report = evaluate_article("source", "The count was 50.", "The count was 19.")
    assert report.character_edit_distance == 2
    assert report.human_reference_similarity.availability is Availability.UNAVAILABLE
    assert report.factuality.attribution.availability is Availability.UNAVAILABLE
    assert report.factuality.new_errors_introduced is None
    assert report.integrity.valid
    assert report.usage.llm_calls == 0


def test_reference_similarity_is_not_named_factuality() -> None:
    """Keep human similarity separate from factual grounding.

    Returns:
        None.
    """
    report = evaluate_article("source", "old", "new", human_reference="new")
    assert report.human_reference_similarity.value == 1.0
    assert report.factuality.attribution.value is None


def test_annotation_metrics_match_factual_phrase_not_location() -> None:
    """Match an issue through the annotated error phrase.

    Returns:
        None.
    """
    annotations = [
        AnnotationLabel(
            annotation_id="ann-1",
            error_text="at least 50 people",
            correction_text="at least 19 people",
            location="paragraph 1",
        )
    ]
    issues = [DetectedIssue(issue_id="issue-1", claim_text="at least 50 people")]
    report = evaluate_article(
        "source",
        "At least 50 people died.",
        "At least 19 people died.",
        annotations=annotations,
        detected_issues=issues,
    )
    assert report.annotation_detection_recall.value == 1.0
    assert report.correction_recall.value == 1.0


def test_edit_precision_requires_independent_labels() -> None:
    """Exclude unreviewed patches from edit precision.

    Returns:
        None.
    """
    patches = [
        PatchObservation(
            patch_id="p1",
            original_text="50",
            replacement_text="19",
            applied=True,
            justified=True,
        ),
        PatchObservation(
            patch_id="p2",
            original_text="said",
            replacement_text="stated",
            applied=False,
            justified=False,
        ),
        PatchObservation(
            patch_id="p3",
            original_text="city",
            replacement_text="state",
            applied=True,
        ),
    ]
    report = evaluate_article("source", "original", "rectified", patches=patches)
    assert report.proposed_edit_precision.value == pytest.approx(0.5)
    assert report.applied_edit_precision.value == 1.0
    assert report.patch_counts.proposed == 3
    assert report.patch_counts.applied == 2


@pytest.mark.parametrize(
    ("rectified", "violation"),
    [
        ("**Error Annotations:**\n[]", "contains_error_annotations"),
        ("```text\narticle\n```", "contains_code_fence"),
        ('{"article": "text"}', "looks_like_json"),
    ],
)
def test_output_integrity_violations(rectified: str, violation: str) -> None:
    """Detect forbidden output artifacts.

    Args:
        rectified: Invalid output text.
        violation: Expected violation code.

    Returns:
        None.
    """
    report = evaluate_article("source", "original", rectified)
    assert not report.integrity.valid
    assert violation in report.integrity.violations
