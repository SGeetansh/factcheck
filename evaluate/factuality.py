"""Interfaces for independently validated factual grounding metrics."""

from evaluate.models import FactualityResult
from evaluate.preservation import unavailable_score


def unavailable_factuality() -> FactualityResult:
    """Return an honest unavailable result when no validated judge is configured.

    Returns:
        Factuality fields without invented scores or error counts.
    """
    return FactualityResult(
        attribution=unavailable_score(
            "source_attribution",
            "No fixed and validated NLI evaluator is configured.",
        ),
        new_errors_introduced=None,
        method=None,
    )
