"""Exact source-evidence contracts."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrievalStrategy(StrEnum):
    """Identify the deterministic method used to retrieve a source passage."""

    BM25 = "bm25"
    SOURCE_CENTRIC = "source_centric"
    EXACT_ANCHOR = "exact_anchor"


class EvidenceSpan(BaseModel):
    """Reference one verbatim passage and its offsets in a source article."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str = Field(min_length=1)
    source_article_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    retrieval_score: float = Field(ge=0.0)
    retrieval_strategy: RetrievalStrategy

    @model_validator(mode="after")
    def validate_span_length(self) -> "EvidenceSpan":
        """Require evidence offsets to match the passage length.

        Returns:
            The validated evidence span.

        Raises:
            ValueError: If offsets do not match the evidence text.
        """
        if self.end - self.start != len(self.text):
            raise ValueError("Evidence offsets must match text length")
        return self

    def verify_against(self, source_text: str) -> bool:
        """Check that the evidence is an exact source substring.

        Args:
            source_text: Complete authoritative source article.

        Returns:
            True when the offsets resolve to the stored passage.
        """
        return source_text[self.start : self.end] == self.text
