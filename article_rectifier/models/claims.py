"""Fact-planning contracts and error classifications."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ErrorType(StrEnum):
    """Classify factual errors without collapsing distinct failure modes."""

    NAMED_ENTITY = "named_entity"
    GEOGRAPHIC = "geographic"
    TEMPORAL = "temporal"
    NUMERICAL = "numerical"
    MAGNITUDE = "magnitude"
    RELATIONSHIP = "relationship"
    NEGATION = "negation"
    ATTRIBUTION = "attribution"
    QUOTATION = "quotation"
    CAUSAL = "causal"
    UNSUPPORTED = "unsupported"
    OTHER = "other"


class FactCheckRequest(BaseModel):
    """Ask one bounded question about an exact claim in one sentence."""

    model_config = ConfigDict(frozen=True)

    request_id: str = Field(min_length=1)
    sentence_id: str = Field(min_length=1)
    claim_text: str = Field(min_length=1)
    verification_question: str = Field(min_length=1)
    search_query: str = Field(min_length=1)
    error_category_hints: tuple[ErrorType, ...] = ()
    sentence_start: int = Field(ge=0)
    sentence_end: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_location(self) -> "FactCheckRequest":
        """Require a non-empty original sentence location.

        Returns:
            The validated request.

        Raises:
            ValueError: If the end offset does not follow the start offset.
        """
        if self.sentence_end <= self.sentence_start:
            raise ValueError("Fact-check sentence range must be non-empty")
        return self


class FactCheckPlan(BaseModel):
    """Contain all unique fact-check requests for one article batch."""

    model_config = ConfigDict(frozen=True)

    requests: tuple[FactCheckRequest, ...] = ()

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "FactCheckPlan":
        """Reject duplicate request identifiers.

        Returns:
            The validated plan.

        Raises:
            ValueError: If request IDs repeat.
        """
        identifiers = [request.request_id for request in self.requests]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Fact-check request IDs must be unique")
        return self
