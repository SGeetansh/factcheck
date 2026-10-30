"""Deterministic sentence segmentation with immutable character offsets."""

from __future__ import annotations

import re

from article_rectifier.models.article import SentenceSpan

ABBREVIATIONS = frozenset(
    {
        "dr",
        "e.g",
        "etc",
        "i.e",
        "mr",
        "mrs",
        "ms",
        "prof",
        "sr",
        "st",
        "u.s",
        "vs",
    }
)
TERMINATORS = frozenset(".!?。\uff01\uff1f")
CLOSING_PUNCTUATION = frozenset("\"'\u2019”)]}")


def _is_decimal(text: str, index: int) -> bool:
    """Check whether a period separates two digits.

    Args:
        text: Paragraph text.
        index: Period position.

    Returns:
        True when digits occur directly before and after the period.
    """
    return (
        text[index] == "."
        and index > 0
        and index + 1 < len(text)
        and text[index - 1].isdigit()
        and text[index + 1].isdigit()
    )


def _is_abbreviation(text: str, index: int) -> bool:
    """Check whether a period ends a known abbreviation or initial.

    Args:
        text: Paragraph text.
        index: Period position.

    Returns:
        True when the preceding token should not terminate a sentence.
    """
    prefix = text[: index + 1]
    token_match = re.search(r"([A-Za-z](?:[A-Za-z.]*)?)\.$", prefix)
    if token_match is None:
        return False
    token = token_match.group(1)
    return token.casefold() in ABBREVIATIONS or (len(token) == 1 and token.isupper())


def _boundary_end(text: str, index: int) -> int | None:
    """Resolve a terminal punctuation position to an exclusive sentence end.

    Args:
        text: Paragraph text.
        index: Candidate punctuation index.

    Returns:
        Exclusive sentence end, or None when the candidate is not a boundary.
    """
    if text[index] == "." and (
        _is_decimal(text, index) or _is_abbreviation(text, index)
    ):
        return None
    end = index + 1
    while end < len(text) and text[end] in CLOSING_PUNCTUATION:
        end += 1
    if end < len(text) and not text[end].isspace():
        return None
    return end


def _paragraph_ranges(text: str) -> list[tuple[int, int]]:
    """Find non-empty paragraph ranges separated by blank lines.

    Args:
        text: Sanitized generated article.

    Returns:
        Ordered non-whitespace paragraph offsets.
    """
    ranges: list[tuple[int, int]] = []
    for match in re.finditer(
        r"(?:^|\n[ \t]*\n)(.*?)(?=\n[ \t]*\n|\Z)", text, re.DOTALL
    ):
        block = match.group(1)
        leading = len(block) - len(block.lstrip())
        trailing = len(block.rstrip())
        if trailing > leading:
            start = match.start(1) + leading
            ranges.append((start, match.start(1) + trailing))
    return ranges


def _split_paragraph(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Split one paragraph into sentence ranges.

    Args:
        text: Complete sanitized article.
        start: Inclusive paragraph start.
        end: Exclusive paragraph end.

    Returns:
        Ordered sentence ranges with surrounding whitespace excluded.
    """
    paragraph = text[start:end]
    if paragraph.lstrip().startswith("#") and "\n" not in paragraph:
        return [(start, end)]
    ranges: list[tuple[int, int]] = []
    local_start = 0
    for index, character in enumerate(paragraph):
        if character not in TERMINATORS:
            continue
        local_end = _boundary_end(paragraph, index)
        if local_end is None:
            continue
        ranges.append((start + local_start, start + local_end))
        local_start = local_end
        while local_start < len(paragraph) and paragraph[local_start].isspace():
            local_start += 1
    if local_start < len(paragraph):
        ranges.append((start + local_start, end))
    return ranges


def split_sentences(text: str) -> list[SentenceSpan]:
    """Split sanitized text into stable sentence spans.

    Args:
        text: Sanitized generated article.

    Returns:
        Ordered sentence spans with offsets into the unchanged input.
    """
    spans: list[SentenceSpan] = []
    for paragraph_index, (start, end) in enumerate(_paragraph_ranges(text), start=1):
        paragraph_id = f"p{paragraph_index:04d}"
        for sentence_start, sentence_end in _split_paragraph(text, start, end):
            spans.append(
                SentenceSpan(
                    sentence_id=f"s{len(spans) + 1:04d}",
                    text=text[sentence_start:sentence_end],
                    start=sentence_start,
                    end=sentence_end,
                    paragraph_id=paragraph_id,
                )
            )
    return spans


def reconstruct_from_spans(text: str, spans: list[SentenceSpan]) -> str:
    """Reconstruct text and validate that spans preserve every non-space token.

    Args:
        text: Original sanitized article.
        spans: Ordered sentence spans derived from the article.

    Returns:
        An exact reconstruction of the input text.

    Raises:
        ValueError: If spans overlap, mismatch, or leave non-whitespace gaps.
    """
    pieces: list[str] = []
    cursor = 0
    for span in spans:
        if span.start < cursor or text[span.start : span.end] != span.text:
            raise ValueError("Sentence spans do not match the original text")
        gap = text[cursor : span.start]
        if gap.strip():
            raise ValueError("Sentence spans leave non-whitespace content uncovered")
        pieces.extend((gap, span.text))
        cursor = span.end
    tail = text[cursor:]
    if tail.strip():
        raise ValueError("Sentence spans leave trailing content uncovered")
    pieces.append(tail)
    return "".join(pieces)
