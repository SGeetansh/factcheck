"""Offline tests for the centralized LiteLLM adapter."""

from pathlib import Path

from pydantic import BaseModel

from article_rectifier.config import Settings
from article_rectifier.llm import LLMClient, LLMResponseStatus, MockLLMTransport


class Answer(BaseModel):
    """Small structured response used by client tests."""

    answer: str


def response(content: str, cost: float | None = None) -> dict[str, object]:
    """Build a LiteLLM-shaped test response.

    Args:
        content: Assistant message content.
        cost: Optional provider-reported cost.

    Returns:
        Dynamic response mapping accepted by the adapter.
    """
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        "_hidden_params": {"response_cost": cost},
    }


def settings(tmp_path: Path, **updates: object) -> Settings:
    """Create isolated test settings.

    Args:
        tmp_path: Temporary project root.
        **updates: Setting overrides.

    Returns:
        Validated settings with a non-secret test key.
    """
    values: dict[str, object] = {
        "LLM_API_KEY": "test-key",
        "project_root": tmp_path,
        "max_llm_retries": 0,
    }
    values.update(updates)
    return Settings(_env_file=None, **values)


def test_valid_json_is_parsed_and_usage_is_recorded(tmp_path: Path) -> None:
    """Validate exact JSON and record observed provider metadata.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    transport = MockLLMTransport([response('{"answer":"West Bengal"}', 0.01)])
    client = LLMClient(settings(tmp_path), transport)
    result = client.complete_structured(
        [{"role": "user", "content": "Question"}],
        Answer,
        prompt_version="test-v1",
    )
    assert result.value == Answer(answer="West Bengal")
    assert result.record.status is LLMResponseStatus.SUCCESS
    assert result.record.input_tokens == 10
    assert result.record.output_tokens == 3
    assert result.record.provider_cost_usd == 0.01
    assert client.provider_cost_usd == 0.01


def test_cache_prevents_a_second_transport_call(tmp_path: Path) -> None:
    """Revalidate cached JSON and avoid duplicate paid requests.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    transport = MockLLMTransport([response('{"answer":"19"}')])
    client = LLMClient(settings(tmp_path), transport)
    arguments = ([{"role": "user", "content": "Count?"}], Answer)
    first = client.complete_structured(*arguments, prompt_version="test-v1")
    second = client.complete_structured(*arguments, prompt_version="test-v1")
    assert first.succeeded and second.succeeded
    assert second.record.status is LLMResponseStatus.CACHE_HIT
    assert len(transport.calls) == 1


def test_invalid_json_gets_one_focused_repair(tmp_path: Path) -> None:
    """Retry malformed output once and accept only validated repair JSON.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    transport = MockLLMTransport(
        [response("West Bengal"), response('{"answer":"West Bengal"}')]
    )
    client = LLMClient(settings(tmp_path), transport)
    result = client.complete_structured(
        [{"role": "user", "content": "Question"}],
        Answer,
        prompt_version="test-v1",
    )
    assert result.succeeded
    assert result.record.retry_count == 1
    repair_messages = transport.calls[1]["messages"]
    assert isinstance(repair_messages, list)
    assert "failed schema validation" in repair_messages[-1]["content"]


def test_repeated_invalid_json_returns_typed_failure(tmp_path: Path) -> None:
    """Return no value instead of coercing arbitrary prose into a correction.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    transport = MockLLMTransport([response("bad"), response("still bad")])
    client = LLMClient(settings(tmp_path), transport)
    result = client.complete_structured(
        [{"role": "user", "content": "Question"}],
        Answer,
        prompt_version="test-v1",
    )
    assert result.value is None
    assert result.record.status is LLMResponseStatus.INVALID_RESPONSE
    assert result.record.error_code == "schema_validation"


def test_empty_response_is_not_repaired_or_invented(tmp_path: Path) -> None:
    """Return an explicit empty-response status without a fabricated value.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    transport = MockLLMTransport([response("  ")])
    client = LLMClient(settings(tmp_path), transport)
    result = client.complete_structured(
        [{"role": "user", "content": "Question"}],
        Answer,
        prompt_version="test-v1",
    )
    assert not result.succeeded
    assert result.record.status is LLMResponseStatus.EMPTY_RESPONSE


def test_cache_key_changes_with_prompt_version(tmp_path: Path) -> None:
    """Keep prompt versions isolated in the deterministic response cache.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    transport = MockLLMTransport(
        [response('{"answer":"one"}'), response('{"answer":"two"}')]
    )
    client = LLMClient(settings(tmp_path), transport)
    messages = [{"role": "user", "content": "Question"}]
    one = client.complete_structured(messages, Answer, prompt_version="v1")
    two = client.complete_structured(messages, Answer, prompt_version="v2")
    assert one.value != two.value
    assert len(transport.calls) == 2


def test_api_key_is_not_stored_in_call_record(tmp_path: Path) -> None:
    """Keep credentials out of observable request metadata.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    client = LLMClient(
        settings(tmp_path),
        MockLLMTransport([response('{"answer":"ok"}')]),
    )
    result = client.complete_structured(
        [{"role": "user", "content": "Question"}],
        Answer,
        prompt_version="v1",
    )
    assert "test-key" not in result.record.model_dump_json()


def test_observed_run_cost_blocks_later_uncached_call(tmp_path: Path) -> None:
    """Stop new transport calls after the configured observed-cost limit.

    Args:
        tmp_path: Temporary cache root.

    Returns:
        None.
    """
    transport = MockLLMTransport([response('{"answer":"one"}', 0.02)])
    client = LLMClient(settings(tmp_path, max_run_cost_usd=0.01), transport)
    first = client.complete_structured(
        [{"role": "user", "content": "One"}], Answer, prompt_version="v1"
    )
    second = client.complete_structured(
        [{"role": "user", "content": "Two"}], Answer, prompt_version="v1"
    )
    assert first.succeeded
    assert not second.succeeded
    assert second.record.status is LLMResponseStatus.BUDGET_BLOCKED
    assert len(transport.calls) == 1
