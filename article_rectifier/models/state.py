"""Typed LangGraph state definitions."""

from typing import TypedDict

from article_rectifier.llm.client import LLMCallRecord
from article_rectifier.models.article import ArticleBatch, SentenceSpan
from article_rectifier.models.claims import FactCheckRequest
from article_rectifier.models.edits import SurgicalEdit
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import PatchResult, VerificationResult
from article_rectifier.retrieval.source_index import SourceIndex


class BatchState(TypedDict):
    """Track immutable inputs and bounded outputs for one article batch."""

    article_id: str
    run_id: str
    batch: ArticleBatch
    source_text: str
    source_index: SourceIndex
    original_text: str
    fact_requests: list[FactCheckRequest]
    retrieved_evidence: dict[str, list[EvidenceSpan]]
    verification_results: list[VerificationResult]
    proposed_edits: list[SurgicalEdit]
    patch_results: list[PatchResult]
    patch_history: list[PatchResult]
    llm_records: list[LLMCallRecord]
    retry_count: int
    status: str
    candidate_text: str
    errors: list[str]


class ArticleState(TypedDict):
    """Track immutable article data and mergeable batch outcomes."""

    article_id: str
    source_hash: str
    original_hash: str
    source_text: str
    sanitized_original: str
    sentences: list[SentenceSpan]
    batches: list[ArticleBatch]
    accepted_edits: list[SurgicalEdit]
    rejected_patches: list[PatchResult]
    unresolved_issues: list[VerificationResult]
    final_output: str
    status: str
    llm_calls: int
    input_tokens: int
    output_tokens: int
