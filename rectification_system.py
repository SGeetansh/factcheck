"""Thin compatibility entry point for the source-grounded pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel

from article_rectifier.config import Settings, load_settings
from article_rectifier.llm.client import LLMCallRecord, LLMClient
from article_rectifier.models.article import ArticleInput
from article_rectifier.observability.logging import AuditLogger
from article_rectifier.pipeline import ArticleRectificationPipeline, PipelineResult


class SmokeTestResponse(BaseModel):
    """Require one exact structured response from a connection smoke test."""

    status: Literal["ok"]


BudgetCheck = Callable[[Settings], float]


def check_provider_budget(settings: Settings) -> float:
    """Read actual proxy key balance with the supplied budget utility.

    Args:
        settings: Validated live endpoint configuration.

    Returns:
        Verified remaining provider balance in USD.

    Raises:
        RuntimeError: If the proxy does not report a finite key budget.
    """
    from budget_checker import get_key_info

    key = settings.require_api_key().get_secret_value()
    info = get_key_info(key, settings.llm_api_base)
    data = info.get("info") or info
    maximum = data.get("max_budget")
    spend = data.get("spend")
    if maximum is None or spend is None:
        raise RuntimeError("Provider balance could not be verified")
    return float(maximum) - float(spend)


def smoke_test_connection(
    settings: Settings | None = None,
    *,
    llm_client: LLMClient | None = None,
    budget_check: BudgetCheck = check_provider_budget,
) -> LLMCallRecord:
    """Make one explicit structured connection call with budget checks.

    Args:
        settings: Optional validated endpoint configuration.
        llm_client: Optional mocked or configured centralized adapter.
        budget_check: Callable that verifies actual balance before and after.

    Returns:
        Observable call record for the accepted model identifier.

    Raises:
        RuntimeError: If balance is insufficient or the response is invalid.
    """
    configured = settings or load_settings()
    remaining = budget_check(configured)
    if remaining <= configured.minimum_budget_reserve_usd:
        raise RuntimeError("Insufficient verified provider budget for smoke test")
    client = llm_client or LLMClient(configured)
    result = client.complete_structured(
        [
            {
                "role": "user",
                "content": 'Return only this JSON object: {"status":"ok"}',
            }
        ],
        SmokeTestResponse,
        prompt_version="connection-smoke-v1",
    )
    budget_check(configured)
    if result.value is None:
        raise RuntimeError(f"Connection smoke test failed: {result.record.status}")
    return result.record


def build_pipeline(settings: Settings | None = None) -> ArticleRectificationPipeline:
    """Build one reusable configured pipeline without making a model call.

    Args:
        settings: Optional validated settings override.

    Returns:
        Reusable article rectification pipeline.
    """
    configured = settings or load_settings()
    audit = AuditLogger(configured.audit_directory / "rectification.jsonl")
    return ArticleRectificationPipeline(configured, LLMClient(configured), audit)


def run_with_result(
    ai_generated_content: str,
    *,
    source_content: str,
    article_id: str,
    pipeline: ArticleRectificationPipeline | None = None,
) -> PipelineResult:
    """Run source-grounded rectification and return full observable results.

    Args:
        ai_generated_content: Generated article that may contain annotations.
        source_content: Supplied authoritative source article.
        article_id: Stable article identity.
        pipeline: Optional reusable or mocked pipeline.

    Returns:
        Final article text and complete processing metadata.
    """
    service = pipeline or build_pipeline()
    article = ArticleInput(
        article_id=article_id,
        source_text=source_content,
        generated_text=ai_generated_content,
    )
    return service.rectify(article)


def run(
    ai_generated_content: str,
    *,
    source_content: str,
    article_id: str,
    pipeline: ArticleRectificationPipeline | None = None,
) -> str:
    """Return only rectified article text for starter-code compatibility.

    Args:
        ai_generated_content: Generated article that may contain annotations.
        source_content: Supplied authoritative source article.
        article_id: Stable article identity.
        pipeline: Optional reusable or mocked pipeline.

    Returns:
        Validated article-only output text.
    """
    return run_with_result(
        ai_generated_content,
        source_content=source_content,
        article_id=article_id,
        pipeline=pipeline,
    ).output_text
