"""Regression coverage for actual transport accounting and debug diagnostics."""

import json
from pathlib import Path

import httpx
from openai import PermissionDeniedError
from pydantic import BaseModel

from article_rectifier.config import Settings
from article_rectifier.llm import LLMClient, MockLLMTransport
from article_rectifier.observability.usage import summarize_usage


class Answer(BaseModel):
    """Require an explicit answer for transport tests."""

    answer: str


def reply(content: str) -> dict[str, object]:
    """Return response content with known token usage and cost."""
    return {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 3},
        "_hidden_params": {"response_cost": 0.01},
    }


def test_repairs_count_every_paid_attempt_and_cache_counts_zero(tmp_path: Path) -> None:
    """Include invalid responses in totals; never charge a cache hit as a call."""
    settings = Settings(LLM_API_KEY="test-key", project_root=tmp_path, _env_file=None)
    client = LLMClient(
        settings, MockLLMTransport([reply("bad"), reply('{"answer":"ok"}')])
    )
    args = ([{"role": "user", "content": "Question"}], Answer)
    live = client.complete_structured(*args, prompt_version="test")
    cached = client.complete_structured(*args, prompt_version="test")
    usage = summarize_usage([live.record, cached.record])
    assert usage.llm_calls == 2
    assert usage.input_tokens == 20
    assert usage.output_tokens == 6
    assert usage.provider_cost_usd == 0.02
    assert cached.record.transport_calls == 0
    events = [
        json.loads(line)
        for path in (tmp_path / "audit_logs/calls").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    assert (
        len([event for event in events if event["stage"] == "llm_call_finished"]) == 2
    )
    assert "test-key" not in json.dumps(events)


def test_nonretryable_provider_error_is_recorded(tmp_path: Path) -> None:
    """Keep a permission error observable without retrying or losing the article."""
    settings = Settings(LLM_API_KEY="test-key", project_root=tmp_path, _env_file=None)
    error = PermissionDeniedError(
        "Denied",
        response=httpx.Response(
            403, request=httpx.Request("POST", "https://example.test")
        ),
        body=None,
    )
    client = LLMClient(settings, MockLLMTransport([error]))
    result = client.complete_structured([], Answer, prompt_version="test")
    assert result.record.status == "transport_failure"
    assert result.record.transport_calls == 1
    assert result.record.error_code == "PermissionDeniedError"


def test_valid_response_is_not_logged_as_parse_failure(tmp_path: Path) -> None:
    """Do not create misleading parse-failure events for successful JSON."""
    settings = Settings(
        LLM_API_KEY="test-key",
        RECTIFIER_DEBUG=True,
        project_root=tmp_path,
        _env_file=None,
    )
    client = LLMClient(settings, MockLLMTransport([reply('{"answer":"ok"}')]))
    assert client.complete_structured([], Answer, prompt_version="test").succeeded
    assert not list((tmp_path / "audit_logs/debug").glob("*.jsonl"))
