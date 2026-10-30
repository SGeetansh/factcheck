"""Offline tests for the explicitly invoked connection smoke test."""

from pathlib import Path

from article_rectifier.config import Settings
from article_rectifier.llm import LLMClient, LLMResponseStatus, MockLLMTransport
from rectification_system import smoke_test_connection


def test_smoke_test_checks_budget_before_and_after(tmp_path: Path) -> None:
    """Validate one model identifier only between two actual-balance checks.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        max_llm_retries=0,
        _env_file=None,
    )
    transport = MockLLMTransport(
        [{"choices": [{"message": {"content": '{"status":"ok"}'}}]}]
    )
    balances = iter((2.78, 2.77))
    observed: list[float] = []

    def budget_check(_: Settings) -> float:
        """Return sequential verified test balances.

        Args:
            _: Unused validated settings.

        Returns:
            Next test balance.
        """
        balance = next(balances)
        observed.append(balance)
        return balance

    record = smoke_test_connection(
        settings,
        llm_client=LLMClient(settings, transport),
        budget_check=budget_check,
    )
    assert record.status is LLMResponseStatus.SUCCESS
    assert observed == [2.78, 2.77]
    assert len(transport.calls) == 1


def test_smoke_test_blocks_call_when_verified_budget_is_low(tmp_path: Path) -> None:
    """Avoid a model request when actual remaining balance is below reserve.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        _env_file=None,
    )
    transport = MockLLMTransport([])
    try:
        smoke_test_connection(
            settings,
            llm_client=LLMClient(settings, transport),
            budget_check=lambda _: 0.10,
        )
    except RuntimeError as error:
        assert "Insufficient" in str(error)
    else:
        raise AssertionError("Low verified balance did not block smoke test")
    assert transport.calls == []
