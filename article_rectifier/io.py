"""Repository-relative mapping, input loading, and atomic output writing."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from article_rectifier.config import PROJECT_ROOT


class ArticleMapping(BaseModel):
    """Map one stable article ID to supplied input and required output files."""

    model_config = ConfigDict(frozen=True)

    article_id: str = Field(pattern=r"^article_\d{3}$")
    source_file: Path
    ai_generated_file: Path
    rectified_file: Path


class MappingCollection(BaseModel):
    """Validate uniqueness and completeness of all article mappings."""

    model_config = ConfigDict(frozen=True)

    articles: tuple[ArticleMapping, ...]

    @model_validator(mode="after")
    def validate_unique_article_ids(self) -> MappingCollection:
        """Reject duplicate article identifiers.

        Returns:
            The validated mapping collection.

        Raises:
            ValueError: If an article ID appears more than once.
        """
        identifiers = [item.article_id for item in self.articles]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Article mapping IDs must be unique")
        return self


def load_article_mappings(
    project_root: Path = PROJECT_ROOT,
) -> tuple[ArticleMapping, ...]:
    """Load and validate the repository article mapping.

    Args:
        project_root: Repository root containing `article_mapping.json`.

    Returns:
        Ordered validated article mappings.
    """
    path = project_root / "article_mapping.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    articles = tuple(ArticleMapping.model_validate(item) for item in data)
    return MappingCollection(articles=articles).articles


def mapping_by_id(
    article_id: str,
    project_root: Path = PROJECT_ROOT,
) -> ArticleMapping:
    """Find one validated article mapping.

    Args:
        article_id: Required stable article identity.
        project_root: Repository root containing the mapping file.

    Returns:
        Matching article mapping.

    Raises:
        ValueError: If the article ID is absent.
    """
    for mapping in load_article_mappings(project_root):
        if mapping.article_id == article_id:
            return mapping
    raise ValueError(f"Article {article_id} not found in mapping")


def read_mapped_text(path: Path, project_root: Path = PROJECT_ROOT) -> str:
    """Read one UTF-8 text path relative to the repository root.

    Args:
        path: Relative or absolute mapped file path.
        project_root: Repository root for relative paths.

    Returns:
        Exact decoded file content.
    """
    resolved = path if path.is_absolute() else project_root / path
    return resolved.read_text(encoding="utf-8")


def output_path_for(
    mapping: ArticleMapping,
    project_root: Path = PROJECT_ROOT,
    output_directory: Path | None = None,
) -> Path:
    """Resolve the required or test-specific output path.

    Args:
        mapping: Validated article mapping.
        project_root: Repository root for the default mapped output.
        output_directory: Optional safe dedicated test output directory.

    Returns:
        Absolute output path with the required original filename.
    """
    if output_directory is not None:
        return output_directory.resolve() / mapping.rectified_file.name
    path = mapping.rectified_file
    return path if path.is_absolute() else project_root / path


def atomic_write_text(destination: Path, content: str) -> None:
    """Write UTF-8 article text atomically in its destination directory.

    Args:
        destination: Final output file path.
        content: Article-only output text.

    Returns:
        None.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
