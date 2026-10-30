"""Article, sentence, and batch data contracts."""

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ArticleInput(BaseModel):
    """Contain immutable text inputs for one source-grounded run."""

    model_config = ConfigDict(frozen=True)

    article_id: str = Field(min_length=1)
    source_text: str
    generated_text: str


class SentenceSpan(BaseModel):
    """Map one deterministic sentence to the sanitized article text."""

    model_config = ConfigDict(frozen=True)

    sentence_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    paragraph_id: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_span_length(self) -> "SentenceSpan":
        """Require offsets to describe the exact sentence length.

        Returns:
            The validated sentence span.

        Raises:
            ValueError: If offsets are reversed or have a different length.
        """
        if self.end < self.start:
            raise ValueError("Sentence end must not precede start")
        if self.end - self.start != len(self.text):
            raise ValueError("Sentence offsets must match text length")
        return self


class ArticleBatch(BaseModel):
    """Contain editable sentences and adjacent read-only context."""

    model_config = ConfigDict(frozen=True)

    batch_id: str = Field(min_length=1)
    sentences: tuple[SentenceSpan, ...] = Field(min_length=1, max_length=10)
    context_before: SentenceSpan | None = None
    context_after: SentenceSpan | None = None

    @property
    def editable_sentence_ids(self) -> tuple[str, ...]:
        """Return sentence IDs in their immutable article order.

        Returns:
            Ordered editable sentence IDs.
        """
        return tuple(sentence.sentence_id for sentence in self.sentences)
