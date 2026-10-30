"""Tests for deterministic preservation metrics."""

import pytest

from evaluate.preservation import (
    changed_character_ratio,
    levenshtein_distance,
    normalized_similarity,
)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [("", "", 0), ("kitten", "sitting", 3), ("West Bengal", "Karnataka", 10)],
)
def test_levenshtein_distance(left: str, right: str, expected: int) -> None:
    """Calculate known edit distances.

    Args:
        left: First test text.
        right: Second test text.
        expected: Known distance.

    Returns:
        None.
    """
    assert levenshtein_distance(left, right) == expected


def test_preservation_uses_original_length() -> None:
    """Use the original length as the required denominator.

    Returns:
        None.
    """
    result = normalized_similarity("abcd", "abXd", "preservation")
    assert result.value == pytest.approx(0.75)
    assert result.numerator == 1
    assert result.denominator == 4


@pytest.mark.parametrize(
    ("original", "candidate", "expected"),
    [("", "", 1.0), ("", "text", 0.0), ("text", "", 0.0)],
)
def test_empty_preservation_rules(
    original: str,
    candidate: str,
    expected: float,
) -> None:
    """Handle empty texts explicitly.

    Args:
        original: Baseline text.
        candidate: Compared text.
        expected: Required score.

    Returns:
        None.
    """
    result = normalized_similarity(original, candidate, "preservation")
    assert result.value == expected


def test_changed_character_ratio_can_exceed_one() -> None:
    """Report the raw ratio instead of silently clipping it.

    Returns:
        None.
    """
    assert changed_character_ratio("a", "abcd") == 3.0
