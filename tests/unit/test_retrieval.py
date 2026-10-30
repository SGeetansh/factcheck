"""Tests for deterministic exact-source evidence retrieval."""

import pytest

from article_rectifier.models import RetrievalStrategy
from article_rectifier.retrieval import build_source_index, search_source

SOURCE = """Source Article(s):
Published: September 6, 2025

The Bengal Files opened in cinemas on September 5, 2025.

Most theater owners in West Bengal refused to screen the film.

Director Mira Sen said the dispute did not delay the national release.

The event drew 19 people and raised ₹3.5 lakh."""


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("Bengal Files theater owners ban", "West Bengal"),
        ("Bengal Files release date", "September 5, 2025"),
        ("event people count", "19 people"),
        ("director national release", "Mira Sen"),
        ("cinemas opening day", "opened in cinemas"),
    ],
)
def test_retrieval_handles_wrong_fact_categories(query: str, expected: str) -> None:
    """Retrieve replacement evidence through stable factual anchors.

    Args:
        query: Query that omits the incorrect generated value.
        expected: Exact correct fact expected in a result.

    Returns:
        None.
    """
    index = build_source_index(SOURCE, "article_004")
    results = search_source(query, index)
    assert results
    assert any(expected in result.text for result in results)


def test_source_centric_fallback_uses_independent_anchors() -> None:
    """Recover when an incorrect query term has no source match.

    Returns:
        None.
    """
    index = build_source_index(SOURCE, "article_004")
    results = search_source(
        "Karnataka",
        index,
        source_anchors=("theater owners", "film"),
    )
    assert results
    assert results[0].retrieval_strategy is RetrievalStrategy.SOURCE_CENTRIC
    assert "West Bengal" in results[0].text


def test_absent_evidence_returns_empty_list() -> None:
    """Do not invent evidence when no query or fallback token occurs.

    Returns:
        None.
    """
    index = build_source_index(SOURCE, "article_004")
    assert search_source("unrelated quantum submarine", index) == []


def test_all_evidence_offsets_match_sanitized_source() -> None:
    """Require every returned passage to be an exact source substring.

    Returns:
        None.
    """
    index = build_source_index(SOURCE, "article_004")
    for evidence in search_source("Bengal film release", index, top_k=10):
        assert index.source_text[evidence.start : evidence.end] == evidence.text
        assert evidence.verify_against(index.source_text)


def test_ranking_is_stable_and_top_k_is_enforced() -> None:
    """Produce identical identifiers and ordering for repeated searches.

    Returns:
        None.
    """
    index = build_source_index(SOURCE, "article_004")
    first = search_source("Bengal film release", index, top_k=2)
    second = search_source("Bengal film release", index, top_k=2)
    assert first == second
    assert len(first) == 2


def test_invalid_top_k_is_rejected() -> None:
    """Reject an invalid evidence limit.

    Returns:
        None.
    """
    index = build_source_index(SOURCE, "article_004")
    with pytest.raises(ValueError, match="top_k"):
        search_source("film", index, top_k=0)
