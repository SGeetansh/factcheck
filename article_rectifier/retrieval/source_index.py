"""Build an immutable lexical index with exact source offsets."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from article_rectifier.preprocessing.sanitizer import sanitize_source_article

TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[.'\u2019-][^\W_]+)*", re.UNICODE)
STOP_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "was",
        "were",
        "what",
        "where",
        "who",
        "with",
    }
)


def tokenize(text: str) -> tuple[str, ...]:
    """Tokenize text for stable case-insensitive lexical matching.

    Args:
        text: Query or source passage.

    Returns:
        Ordered normalized tokens excluding a small fixed stop-word list.
    """
    tokens = (match.group(0).casefold() for match in TOKEN_PATTERN.finditer(text))
    return tuple(token for token in tokens if token not in STOP_WORDS)


@dataclass(frozen=True, slots=True)
class IndexedPassage:
    """Store one exact source passage and its lexical term frequencies."""

    passage_id: str
    text: str
    start: int
    end: int
    tokens: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceIndex:
    """Store all local passages and BM25 statistics for one source article."""

    article_id: str
    source_text: str
    passages: tuple[IndexedPassage, ...]
    document_frequencies: dict[str, int]
    average_length: float

    def inverse_document_frequency(self, token: str) -> float:
        """Calculate BM25 inverse document frequency for one token.

        Args:
            token: Normalized query token.

        Returns:
            Non-negative BM25 inverse document frequency.
        """
        frequency = self.document_frequencies.get(token, 0)
        count = len(self.passages)
        return math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    """Find non-empty source paragraph ranges.

    Args:
        text: Sanitized source text.

    Returns:
        Exact paragraph ranges in source order.
    """
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(r"\S(?:.*?\S)?(?=\n[ \t]*\n|\Z)", text, re.DOTALL):
        ranges.append((match.start(), match.end()))
    return ranges


def _passage_ranges(text: str) -> list[tuple[int, int]]:
    """Create paragraphs and adjacent two-paragraph windows.

    Args:
        text: Sanitized source text.

    Returns:
        Stable de-duplicated passage ranges.
    """
    paragraphs = _paragraph_ranges(text)
    ranges = list(paragraphs)
    ranges.extend(
        (paragraphs[index][0], paragraphs[index + 1][1])
        for index in range(len(paragraphs) - 1)
    )
    return sorted(set(ranges), key=lambda item: (item[0], item[1]))


def _document_frequencies(passages: list[IndexedPassage]) -> dict[str, int]:
    """Count passages containing each lexical token.

    Args:
        passages: Indexed source passages.

    Returns:
        Token-to-document-frequency mapping.
    """
    frequencies: dict[str, int] = {}
    for passage in passages:
        for token in set(passage.tokens):
            frequencies[token] = frequencies.get(token, 0) + 1
    return frequencies


def build_source_index(source_text: str, article_id: str) -> SourceIndex:
    """Build one deterministic in-memory source index.

    Args:
        source_text: Complete supplied source article.
        article_id: Stable source article identifier.

    Returns:
        An immutable index of exact source passages.
    """
    sanitized = sanitize_source_article(source_text)
    passages = [
        IndexedPassage(
            passage_id=f"src-{article_id}-{index:04d}",
            text=sanitized[start:end],
            start=start,
            end=end,
            tokens=tokenize(sanitized[start:end]),
        )
        for index, (start, end) in enumerate(_passage_ranges(sanitized), start=1)
    ]
    average = sum(len(passage.tokens) for passage in passages) / max(1, len(passages))
    return SourceIndex(
        article_id=article_id,
        source_text=sanitized,
        passages=tuple(passages),
        document_frequencies=_document_frequencies(passages),
        average_length=average,
    )
