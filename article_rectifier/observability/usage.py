"""Aggregate observed LLM calls without fabricating missing billing data."""

from pydantic import BaseModel, ConfigDict, Field

from article_rectifier.llm.client import LLMCallRecord


class UsageSummary(BaseModel):
    """Summarize observed calls, tokens, retries, latency, and provider cost."""

    model_config = ConfigDict(frozen=True)

    llm_calls: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    retries: int = Field(ge=0)
    latency_seconds: float = Field(ge=0.0)
    provider_cost_usd: float | None = Field(default=None, ge=0.0)
    provider_cost_records: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)


def summarize_usage(records: list[LLMCallRecord]) -> UsageSummary:
    """Aggregate only values observed in model response metadata.

    Args:
        records: Logical LLM request records from one or more articles.

    Returns:
        Usage summary with provider cost unavailable when no call reported it.
    """
    costs = [
        item.provider_cost_usd for item in records if item.provider_cost_usd is not None
    ]
    return UsageSummary(
        llm_calls=len(records),
        input_tokens=sum(item.input_tokens for item in records),
        output_tokens=sum(item.output_tokens for item in records),
        retries=sum(item.retry_count for item in records),
        latency_seconds=sum(item.latency_seconds for item in records),
        provider_cost_usd=sum(costs) if costs else None,
        provider_cost_records=len(costs),
        estimated_cost_usd=None,
    )
