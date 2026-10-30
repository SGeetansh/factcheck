"""Validated runtime configuration for the rectification pipeline."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Define environment and safety settings for one pipeline process."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    llm_api_key: SecretStr | None = Field(default=None, alias="LLM_API_KEY")
    llm_api_base: str = Field(
        default="https://recllm.brahmastra.tech/",
        alias="LLM_API_BASE",
    )
    llm_model_name: str = Field(
        default="gpt-oss-120b",
        alias="LLM_MODEL_NAME",
    )
    batch_size: int = Field(default=3, ge=1, le=10)
    retrieval_top_k: int = Field(default=4, ge=1, le=20)
    max_patch_retries: int = Field(default=1, ge=0, le=3)
    max_llm_retries: int = Field(default=2, ge=0, le=5)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    enable_validation_llm: bool = True
    batch_verification: bool = True
    soft_patch_character_limit: int = Field(default=120, ge=1, le=1000)
    llm_timeout_seconds: float = Field(default=45.0, gt=0.0, le=300.0)
    max_concurrency: int = Field(default=1, ge=1, le=8)
    max_run_cost_usd: float = Field(default=2.70, gt=0.0, le=3.0)
    minimum_budget_reserve_usd: float = Field(default=0.20, ge=0.0, le=3.0)
    enable_llm_cache: bool = True
    rectifier_cache_only: bool = Field(default=False, alias="RECTIFIER_CACHE_ONLY")
    rectifier_debug: bool = Field(default=False, alias="RECTIFIER_DEBUG")
    use_native_structured_output: bool = False
    enable_llm_body_logging: bool = False
    log_level: str = Field(default="INFO", pattern=r"^(DEBUG|INFO|WARNING|ERROR)$")
    project_root: Path = PROJECT_ROOT

    @property
    def cache_directory(self) -> Path:
        """Return the configured local response-cache directory.

        Returns:
            Path below the project root. The property does not create it.
        """
        return self.project_root / ".cache" / "article_rectifier"

    @property
    def audit_directory(self) -> Path:
        """Return the configured structured-audit directory.

        Returns:
            Path below the project root. The property does not create it.
        """
        return self.project_root / "audit_logs"

    def require_api_key(self) -> SecretStr:
        """Return the configured API key for an explicitly requested live call.

        Returns:
            The protected API key value.

        Raises:
            ValueError: If no API key is configured.
        """
        if self.llm_api_key is None or not self.llm_api_key.get_secret_value():
            raise ValueError("LLM_API_KEY is required for live LLM calls")
        return self.llm_api_key


def load_settings(*, project_root: Path | None = None) -> Settings:
    """Load validated settings without performing network operations.

    Args:
        project_root: Optional repository root override for tests or embedding.

    Returns:
        Validated settings populated from the environment and `.env` file.
    """
    values = {"project_root": project_root} if project_root is not None else {}
    return Settings(**values)
