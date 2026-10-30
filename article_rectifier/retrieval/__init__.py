"""Deterministic local retrieval over one supplied source article."""

from article_rectifier.retrieval.retriever import search_source
from article_rectifier.retrieval.source_index import SourceIndex, build_source_index

__all__ = ["SourceIndex", "build_source_index", "search_source"]
