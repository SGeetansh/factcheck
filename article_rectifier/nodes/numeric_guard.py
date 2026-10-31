"""Compare numeric values deterministically without requiring identical prose."""

import re
from decimal import Decimal

from article_rectifier.models.edits import SurgicalEdit
from article_rectifier.models.evidence import EvidenceSpan

NUMBER = re.compile(
    r"(?<!\w)(\d[\d,]*(?:\.\d+)?)\s*(crore|lakh|million|billion|thousand|l\b)?", re.I
)
SCALE = {
    "crore": 10_000_000,
    "lakh": 100_000,
    "l": 100_000,
    "million": 1_000_000,
    "billion": 1_000_000_000,
    "thousand": 1_000,
}


def numeric_values(text: str) -> set[Decimal]:
    """Extract exact numeric magnitudes, including Indian and Western scales.

    Args:
        text: Original, replacement, or source text.

    Returns:
        Normalized numeric values. Semantic roles are checked by the reviewer.
    """
    values = {
        Decimal(match[1].replace(",", "")) * SCALE.get((match[2] or "").lower(), 1)
        for match in NUMBER.finditer(text)
    }
    words = {
        "zero": 0,
        "one": 1,
        "two": 2,
        "three": 3,
        "four": 4,
        "five": 5,
        "six": 6,
        "seven": 7,
        "eight": 8,
        "nine": 9,
        "ten": 10,
        "eleven": 11,
        "twelve": 12,
    }
    values.update(
        Decimal(words[word])
        for word in re.findall(r"\b\w+\b", text.lower())
        if word in words
    )
    return values


def numeric_error(edit: SurgicalEdit, evidence: list[EvidenceSpan]) -> str | None:
    """Reject introduced numbers absent from the exact supporting passages.

    Args:
        edit: Proposed factual correction.
        evidence: Exact cited source passages.

    Returns:
        Rejection reason, or None. This is not a factual entailment score.
    """
    introduced = numeric_values(edit.replacement_text) - numeric_values(
        edit.original_text
    )
    source_values = numeric_values(" ".join(item.text for item in evidence))
    missing = introduced - source_values
    if missing:
        return "replacement numbers absent from cited evidence: " + ", ".join(
            str(x) for x in sorted(missing)
        )
    return None
