"""Mocked integration tests for compatibility entry points and full output."""

import hashlib
import subprocess
from pathlib import Path

import rectifier
from article_rectifier.models import ArticleInput
from article_rectifier.pipeline import ArticleProcessingStatus, PipelineResult
from article_rectifier.preprocessing.sanitizer import sanitize_generated_article
from rectification_system import run


class UnchangedPipeline:
    """Offline pipeline double that returns sanitized generated text unchanged."""

    def rectify(self, article: ArticleInput) -> PipelineResult:
        """Return a completed deterministic no-call result.

        Args:
            article: Immutable article input.

        Returns:
            Sanitized unchanged article result.
        """
        output = sanitize_generated_article(article.generated_text)
        digest = hashlib.sha256(output.encode()).hexdigest()
        return PipelineResult(
            article_id=article.article_id,
            output_text=output,
            status=ArticleProcessingStatus.COMPLETED,
            accepted_edits=(),
            rejected_patches=(),
            unresolved_issues=(),
            llm_records=(),
            errors=(),
            source_hash=digest,
            original_hash=digest,
        )


class FailingPipeline:
    """Offline pipeline double that raises one unexpected runtime failure."""

    def rectify(self, article: ArticleInput) -> PipelineResult:
        """Raise a test failure for fallback validation.

        Args:
            article: Immutable article input.

        Raises:
            RuntimeError: Always.
        """
        raise RuntimeError(f"failed {article.article_id}")


def test_compatibility_run_delegates_source_and_identity() -> None:
    """Return article-only text through the root compatibility function.

    Returns:
        None.
    """
    output = run(
        "Article text.",
        source_content="Source text.",
        article_id="article_001",
        pipeline=UnchangedPipeline(),  # type: ignore[arg-type]
    )
    assert output == "Article text."


def test_rectify_all_mock_creates_all_104_outputs(tmp_path: Path) -> None:
    """Exercise full-dataset filenames and integrity without real model calls.

    Args:
        tmp_path: Dedicated output directory that protects human references.

    Returns:
        None.
    """
    summary = rectifier.rectify_all(
        pipeline=UnchangedPipeline(),  # type: ignore[arg-type]
        output_directory=tmp_path,
    )
    outputs = sorted(tmp_path.glob("article_*.txt"))
    assert summary.total_articles == 104
    assert summary.completed_articles == 104
    assert summary.usage.llm_calls == 0
    assert len(outputs) == 104
    assert outputs[0].name == "article_001.txt"
    assert outputs[-1].name == "article_104.txt"
    assert all("Error Annotations:" not in item.read_text() for item in outputs)


def test_article_failure_writes_sanitized_degraded_fallback(tmp_path: Path) -> None:
    """Keep output contract after one unexpected pipeline exception.

    Args:
        tmp_path: Dedicated output directory.

    Returns:
        None.
    """
    summary = rectifier.test_rectifier(
        1,
        pipeline=FailingPipeline(),  # type: ignore[arg-type]
        output_directory=tmp_path,
    )
    assert summary.failed_articles == 1
    output = (tmp_path / "article_001.txt").read_text()
    assert output
    assert "Error Annotations:" not in output


def test_cli_help_loads_without_network_or_model_call() -> None:
    """Keep the required root CLI importable and self-documenting.

    Returns:
        None.
    """
    completed = subprocess.run(
        ["uv", "run", "python", "rectifier.py", "--help"],
        cwd=Path(__file__).resolve().parents[2],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "rectify-all" in completed.stdout
    assert "test" in completed.stdout
