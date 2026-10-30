"""Tests for validated pipeline settings."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from article_rectifier.config import Settings, load_settings


def test_settings_have_safe_sequential_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load deterministic defaults without exposing a configured secret.

    Args:
        monkeypatch: Pytest environment mutation helper.

    Returns:
        None.
    """
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    settings = Settings(_env_file=None)
    assert settings.batch_size == 3
    assert settings.max_concurrency == 1
    assert settings.temperature == 0.0
    assert settings.llm_api_key is None


def test_environment_overrides_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Read supported environment fields and reject unsafe values.

    Args:
        monkeypatch: Pytest environment mutation helper.

    Returns:
        None.
    """
    monkeypatch.setenv("BATCH_SIZE", "4")
    assert Settings(_env_file=None).batch_size == 4
    monkeypatch.setenv("BATCH_SIZE", "0")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_secret_is_masked() -> None:
    """Prevent API credentials from appearing in model representations.

    Returns:
        None.
    """
    settings = Settings(LLM_API_KEY="private-key", _env_file=None)
    assert "private-key" not in repr(settings)
    assert settings.require_api_key().get_secret_value() == "private-key"


def test_api_key_is_required_only_for_live_calls() -> None:
    """Allow offline construction and fail on explicit key access.

    Returns:
        None.
    """
    settings = Settings(_env_file=None)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        settings.require_api_key()


def test_project_paths_follow_override(tmp_path: Path) -> None:
    """Resolve writable runtime paths below an injected project root.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    settings = load_settings(project_root=tmp_path)
    assert settings.cache_directory == tmp_path / ".cache" / "article_rectifier"
    assert settings.audit_directory == tmp_path / "audit_logs"
