"""Validated contracts shared by rectification pipeline stages."""

from article_rectifier.models.article import ArticleBatch, ArticleInput, SentenceSpan
from article_rectifier.models.claims import ErrorType, FactCheckPlan, FactCheckRequest
from article_rectifier.models.edits import EditOperation, SurgicalEdit
from article_rectifier.models.evidence import EvidenceSpan, RetrievalStrategy
from article_rectifier.models.results import (
    PatchResult,
    PatchStatus,
    Verdict,
    VerificationResult,
    VerificationStatus,
)

__all__ = [
    "ArticleBatch",
    "ArticleInput",
    "EditOperation",
    "ErrorType",
    "EvidenceSpan",
    "FactCheckPlan",
    "FactCheckRequest",
    "PatchResult",
    "PatchStatus",
    "RetrievalStrategy",
    "SentenceSpan",
    "SurgicalEdit",
    "Verdict",
    "VerificationResult",
    "VerificationStatus",
]
