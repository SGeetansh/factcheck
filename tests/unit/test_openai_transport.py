"""Check OpenAI proxy transport without making real network requests."""

from types import SimpleNamespace

from article_rectifier.llm import openai_transport


def test_openai_transport_uses_proxy_v1_and_raw_model(monkeypatch) -> None:
    """Verify exact URL/model sent to an injected fake SDK client.

    Args:
        monkeypatch: Pytest hook for replacing the OpenAI client.

    Returns:
        None.
    """
    observations: dict[str, object] = {}

    def fake_create(**kwargs: object) -> object:
        """Capture one fake completion request."""
        observations["request"] = kwargs
        return {"choices": [{"message": {"content": '{"ok":true}'}}]}

    def fake_client(**kwargs: object) -> object:
        """Capture OpenAI client settings and provide a fake endpoint."""
        observations["client"] = kwargs
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create))
        )

    monkeypatch.setattr(openai_transport, "OpenAI", fake_client)
    response = openai_transport.openai_completion(
        model="gpt-oss-120b",
        messages=[{"role": "user", "content": "Say hello"}],
        api_key="test-only",
        api_base="https://recllm.brahmastra.tech/",
        temperature=0.0,
        timeout=30.0,
    )
    assert observations["client"]["base_url"] == "https://recllm.brahmastra.tech/v1"
    assert observations["request"]["model"] == "gpt-oss-120b"
    assert response["choices"][0]["message"]["content"] == '{"ok":true}'


def test_openai_transport_does_not_double_v1(monkeypatch) -> None:
    """Avoid appending a duplicate version prefix to a versioned proxy URL.

    Args:
        monkeypatch: Pytest monkeypatch fixture.

    Returns:
        None.
    """
    observed: dict[str, object] = {}

    def fake_client(**kwargs: object) -> object:
        """Return a fake SDK client and capture its settings."""
        observed.update(kwargs)
        return SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: {}))
        )

    monkeypatch.setattr(openai_transport, "OpenAI", fake_client)
    openai_transport.openai_completion(
        model="gpt-oss-120b",
        messages=[],
        api_key="test-only",
        api_base="https://recllm.brahmastra.tech/v1",
        temperature=0.0,
        timeout=30.0,
    )
    assert observed["base_url"] == "https://recllm.brahmastra.tech/v1"
