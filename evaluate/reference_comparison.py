"""Human-reference comparison metrics."""

from evaluate.models import ScoreResult
from evaluate.preservation import normalized_similarity, unavailable_score


def reference_similarity(rectified: str, human_reference: str | None) -> ScoreResult:
    """Compare a rectified article with an optional human reference.

    Args:
        rectified: System output article.
        human_reference: Human-corrected article, when available.

    Returns:
        Assignment-defined reference similarity or an unavailable result.
    """
    if human_reference is None:
        return unavailable_score(
            "human_reference_similarity",
            "No human reference was supplied.",
        )
    return normalized_similarity(
        human_reference,
        rectified,
        "human_reference_similarity",
    )
