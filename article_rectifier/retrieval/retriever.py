"""Rank exact source passages with deterministic BM25 search."""

from __future__ import annotations

from collections import Counter

from article_rectifier.models.evidence import EvidenceSpan, RetrievalStrategy
from article_rectifier.retrieval.source_index import (
    IndexedPassage,
    SourceIndex,
    tokenize,
)


def _bm25_score(
    query_tokens: tuple[str, ...],
    passage: IndexedPassage,
    source_index: SourceIndex,
) -> float:
    """Calculate a BM25 score for one source passage.

    Args:
        query_tokens: Normalized query terms.
        passage: Candidate exact source passage.
        source_index: Index containing corpus statistics.

    Returns:
        Deterministic BM25 relevance score.
    """
    term_counts = Counter(passage.tokens)
    length_ratio = len(passage.tokens) / max(1.0, source_index.average_length)
    score = 0.0
    for token in set(query_tokens):
        frequency = term_counts.get(token, 0)
        if not frequency:
            continue
        numerator = frequency * 2.2
        denominator = frequency + 1.2 * (0.25 + 0.75 * length_ratio)
        score += (
            source_index.inverse_document_frequency(token) * numerator / denominator
        )
    return score


def _rank_passages(
    tokens: tuple[str, ...],
    source_index: SourceIndex,
) -> list[tuple[float, IndexedPassage]]:
    """Rank passages with stable tie-breaking.

    Args:
        tokens: Search terms.
        source_index: Per-article source index.

    Returns:
        Positive-score passages in descending relevance order.
    """
    scored = [
        (_bm25_score(tokens, passage, source_index), passage)
        for passage in source_index.passages
    ]
    positive = [item for item in scored if item[0] > 0.0]
    return sorted(positive, key=lambda item: (-item[0], item[1].start, item[1].end))


def _deduplicate_ranges(
    ranked: list[tuple[float, IndexedPassage]],
    top_k: int,
) -> list[tuple[float, IndexedPassage]]:
    """Prefer compact passages over windows that repeat the same evidence.

    Args:
        ranked: Ranked passage candidates.
        top_k: Maximum passages to retain.

    Returns:
        Non-identical ranked source passages.
    """
    selected: list[tuple[float, IndexedPassage]] = []
    seen_text: set[str] = set()
    for item in ranked:
        normalized = " ".join(item[1].text.split())
        if normalized not in seen_text:
            selected.append(item)
            seen_text.add(normalized)
        if len(selected) == top_k:
            break
    return selected


def search_source(
    query: str,
    source_index: SourceIndex,
    top_k: int = 4,
    *,
    source_anchors: tuple[str, ...] = (),
) -> list[EvidenceSpan]:
    """Retrieve exact passages from the supplied source article.

    Args:
        query: Stable lexical search query.
        source_index: Index of the authoritative source article.
        top_k: Maximum number of passages to return.
        source_anchors: Optional source-centric fallback terms from nearby text.

    Returns:
        Ranked verbatim passages with source offsets. Returns an empty list when
        neither the query nor fallback anchors occur in the source.
    """
    if top_k < 1:
        raise ValueError("top_k must be positive")
    ranked = _rank_passages(tokenize(query), source_index)
    strategy = RetrievalStrategy.BM25
    if not ranked and source_anchors:
        ranked = _rank_passages(tokenize(" ".join(source_anchors)), source_index)
        strategy = RetrievalStrategy.SOURCE_CENTRIC
    evidence: list[EvidenceSpan] = []
    for score, passage in _deduplicate_ranges(ranked, top_k):
        span = EvidenceSpan(
            evidence_id=passage.passage_id,
            source_article_id=source_index.article_id,
            text=passage.text,
            start=passage.start,
            end=passage.end,
            retrieval_score=score,
            retrieval_strategy=strategy,
        )
        if not span.verify_against(source_index.source_text):
            raise ValueError("Retrieved evidence does not match source text")
        evidence.append(span)
    return evidence
