"""Required command-line entry point for article rectification."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

from article_rectifier.config import PROJECT_ROOT
from article_rectifier.io import (
    ArticleMapping,
    atomic_write_text,
    load_article_mappings,
    mapping_by_id,
    output_path_for,
    read_mapped_text,
)
from article_rectifier.observability.usage import UsageSummary, summarize_usage
from article_rectifier.pipeline import (
    ArticleProcessingStatus,
    ArticleRectificationPipeline,
    PipelineResult,
)
from article_rectifier.preprocessing.sanitizer import sanitize_generated_article
from rectification_system import build_pipeline, run_with_result, smoke_test_connection

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class BatchSummary:
    """Summarize accurate article, patch, issue, and usage outcomes."""

    total_articles: int
    completed_articles: int
    degraded_articles: int
    failed_articles: int
    accepted_patches: int
    rejected_patches: int
    unresolved_issues: int
    usage: UsageSummary


def get_article_mapping(article_id: str) -> ArticleMapping:
    """Return one repository-relative validated article mapping.

    Args:
        article_id: Stable identity such as `article_001`.

    Returns:
        Validated mapping for the requested article.
    """
    return mapping_by_id(article_id, PROJECT_ROOT)


def get_ai_generated_article(article_id: str) -> str:
    """Read one exact generated input article.

    Args:
        article_id: Stable article identity.

    Returns:
        Generated article text.
    """
    mapping = get_article_mapping(article_id)
    return read_mapped_text(mapping.ai_generated_file, PROJECT_ROOT)


def get_source_article(article_id: str) -> str:
    """Read one exact authoritative source article.

    Args:
        article_id: Stable article identity.

    Returns:
        Source article text.
    """
    mapping = get_article_mapping(article_id)
    return read_mapped_text(mapping.source_file, PROJECT_ROOT)


def save_rectified_article(
    article_id: str,
    rectified_content: str,
    output_directory: Path | None = None,
) -> Path:
    """Atomically save article-only text under the required filename.

    Args:
        article_id: Stable article identity.
        rectified_content: Validated or degraded fallback article text.
        output_directory: Optional dedicated test output directory.

    Returns:
        Absolute final output path.
    """
    destination = output_path_for(
        get_article_mapping(article_id), PROJECT_ROOT, output_directory
    )
    atomic_write_text(destination, rectified_content)
    return destination


def rectify_article(
    article_id: str,
    pipeline: ArticleRectificationPipeline | None = None,
    output_directory: Path | None = None,
) -> PipelineResult:
    """Rectify one article and save its output atomically.

    Args:
        article_id: Stable article identity.
        pipeline: Optional reusable configured or mocked pipeline.
        output_directory: Optional dedicated test output directory.

    Returns:
        Full observable pipeline result.
    """
    result = run_with_result(
        get_ai_generated_article(article_id),
        source_content=get_source_article(article_id),
        article_id=article_id,
        pipeline=pipeline,
    )
    save_rectified_article(article_id, result.output_text, output_directory)
    return result


def _fallback_result(article_id: str, error: Exception) -> PipelineResult:
    """Create an accurate failed result after an unexpected article failure.

    Args:
        article_id: Failed article identity.
        error: Unexpected exception retained as an error code only.

    Returns:
        Sanitized unchanged article fallback marked failed.
    """
    fallback = sanitize_generated_article(get_ai_generated_article(article_id))
    return PipelineResult(
        article_id=article_id,
        output_text=fallback,
        status=ArticleProcessingStatus.FAILED,
        accepted_edits=(),
        rejected_patches=(),
        unresolved_issues=(),
        llm_records=(),
        errors=(type(error).__name__,),
        source_hash="",
        original_hash="",
    )


def _process_mappings(
    mappings: tuple[ArticleMapping, ...],
    pipeline: ArticleRectificationPipeline,
    output_directory: Path | None,
) -> BatchSummary:
    """Process mappings independently and continue after article failures.

    Args:
        mappings: Ordered article mappings to process.
        pipeline: Reusable configured or mocked pipeline.
        output_directory: Optional dedicated test output directory.

    Returns:
        Accurate aggregate processing summary.
    """
    results: list[PipelineResult] = []
    for index, mapping in enumerate(mappings, start=1):
        LOGGER.info("Processing %s (%s/%s)", mapping.article_id, index, len(mappings))
        try:
            result = rectify_article(mapping.article_id, pipeline, output_directory)
        except Exception as error:
            LOGGER.exception("Article %s failed", mapping.article_id)
            result = _fallback_result(mapping.article_id, error)
            save_rectified_article(
                mapping.article_id, result.output_text, output_directory
            )
        results.append(result)
    return _summarize(results)


def _summarize(results: list[PipelineResult]) -> BatchSummary:
    """Aggregate accurate article and usage observations.

    Args:
        results: Per-article pipeline outcomes.

    Returns:
        Accurate batch summary.
    """
    records = [record for result in results for record in result.llm_records]
    return BatchSummary(
        total_articles=len(results),
        completed_articles=sum(
            item.status is ArticleProcessingStatus.COMPLETED for item in results
        ),
        degraded_articles=sum(
            item.status is ArticleProcessingStatus.DEGRADED for item in results
        ),
        failed_articles=sum(
            item.status is ArticleProcessingStatus.FAILED for item in results
        ),
        accepted_patches=sum(len(item.accepted_edits) for item in results),
        rejected_patches=sum(len(item.rejected_patches) for item in results),
        unresolved_issues=sum(len(item.unresolved_issues) for item in results),
        usage=summarize_usage(records),
    )


def test_rectifier(
    count: int,
    pipeline: ArticleRectificationPipeline | None = None,
    output_directory: Path | None = None,
) -> BatchSummary:
    """Process the first requested mappings for a bounded pilot.

    Args:
        count: Number of ordered articles to process.
        pipeline: Optional reusable configured or mocked pipeline.
        output_directory: Optional dedicated test output directory.

    Returns:
        Accurate batch summary.
    """
    service = pipeline or build_pipeline()
    mappings = load_article_mappings(PROJECT_ROOT)[:count]
    return _process_mappings(mappings, service, output_directory)


def rectify_all(
    pipeline: ArticleRectificationPipeline | None = None,
    output_directory: Path | None = None,
) -> BatchSummary:
    """Process all mapped articles while preserving per-article fallbacks.

    Args:
        pipeline: Optional reusable configured or mocked pipeline.
        output_directory: Optional dedicated test output directory.

    Returns:
        Accurate full-dataset summary.

    Raises:
        RuntimeError: If any required mapped output file is missing.
    """
    service = pipeline or build_pipeline()
    mappings = load_article_mappings(PROJECT_ROOT)
    summary = _process_mappings(mappings, service, output_directory)
    missing = [
        item.article_id
        for item in mappings
        if not output_path_for(item, PROJECT_ROOT, output_directory).is_file()
    ]
    if missing:
        raise RuntimeError(f"Missing output files: {','.join(missing)}")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    """Build the preserved grading CLI and safe test-output option.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser(
        description="Rectify AI-generated articles with supplied source evidence."
    )
    parser.add_argument("command", choices=["test", "rectify-all", "smoke-test"])
    parser.add_argument("--count", type=int, default=16)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def _print_summary(summary: BatchSummary) -> None:
    """Print an accurate compact final summary.

    Args:
        summary: Completed batch observations.
    """
    print(f"Articles: {summary.total_articles}")
    print(f"Completed: {summary.completed_articles}")
    print(f"Degraded: {summary.degraded_articles}")
    print(f"Failed: {summary.failed_articles}")
    print(f"Accepted patches: {summary.accepted_patches}")
    print(f"Rejected patches: {summary.rejected_patches}")
    print(f"Unresolved issues: {summary.unresolved_issues}")
    print(f"LLM calls: {summary.usage.llm_calls}")
    cost = summary.usage.provider_cost_usd
    print(f"Provider-reported cost: {cost if cost is not None else 'unavailable'}")


def main() -> None:
    """Parse CLI arguments and execute the selected preserved command."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _build_parser().parse_args()
    if args.command == "smoke-test":
        record = smoke_test_connection()
        print(f"Connection status: {record.status}")
        print(f"Model: {record.model_identifier}")
        return
    summary = (
        test_rectifier(args.count, output_directory=args.output_dir)
        if args.command == "test"
        else rectify_all(output_directory=args.output_dir)
    )
    _print_summary(summary)


if __name__ == "__main__":
    main()
