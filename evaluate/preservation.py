"""Deterministic edit-distance and preservation metrics."""

from __future__ import annotations

from evaluate.models import Availability, ScoreResult


def levenshtein_distance(left: str, right: str) -> int:
    """Calculate the character Levenshtein distance with bounded memory.

    Args:
        left: First text.
        right: Second text.

    Returns:
        The minimum number of insertions, deletions, and substitutions.
    """
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            insertion = current[right_index - 1] + 1
            deletion = previous[right_index] + 1
            substitution = previous[right_index - 1] + (
                left_character != right_character
            )
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def normalized_similarity(original: str, candidate: str, name: str) -> ScoreResult:
    """Calculate the assignment-defined normalized similarity.

    Args:
        original: Denominator and comparison baseline.
        candidate: Text to compare with the baseline.
        name: Metric name for the result.

    Returns:
        A deterministic score in the inclusive range from zero to one.
    """
    distance = levenshtein_distance(original, candidate)
    if not original:
        value = 1.0 if not candidate else 0.0
        note = "Empty baselines score 1 only when both texts are empty."
    else:
        value = max(0.0, 1.0 - distance / len(original))
        note = None
    return ScoreResult(
        name=name,
        value=value,
        numerator=distance,
        denominator=len(original),
        note=note,
    )


def unavailable_score(name: str, note: str) -> ScoreResult:
    """Create an explicit unavailable score.

    Args:
        name: Metric name.
        note: Reason that the metric is unavailable.

    Returns:
        A score without a fabricated numeric value.
    """
    return ScoreResult(name=name, availability=Availability.UNAVAILABLE, note=note)


def changed_character_ratio(original: str, rectified: str) -> float:
    """Return edit distance divided by original character count.

    Args:
        original: Sanitized generated article.
        rectified: Rectified article.

    Returns:
        The raw Levenshtein distance ratio. The value can exceed one when
        insertions are longer than the original. Two empty texts return zero.
    """
    distance = levenshtein_distance(original, rectified)
    if not original:
        return 0.0 if not rectified else float(distance)
    return distance / len(original)
