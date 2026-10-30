"""OpenAI SDK transport for the provided OpenAI-compatible proxy."""

from __future__ import annotations

from typing import Any

from openai import OpenAI


def openai_completion(
    *,
    model: str,
    messages: list[dict[str, str]],
    api_key: str,
    api_base: str,
    temperature: float,
    timeout: float,
    response_format: dict[str, str] | None = None,
) -> object:
    """Call the proxy with an unprefixed model name.

    Args:
        model: Exact model identifier accepted by the proxy.
        messages: Conversation messages for the model.
        api_key: Private proxy credential.
        api_base: Proxy root URL, with or without a trailing ``/v1``.
        temperature: Requested sampling temperature.
        timeout: Request timeout in seconds.
        response_format: Optional OpenAI-compatible JSON output setting.

    Returns:
        OpenAI ChatCompletion with choices and token usage.
    """
    base_url = api_base.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"
    client = OpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        max_retries=0,  # Retries are controlled by the caller.
    )
    options: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }
    if response_format is not None:
        options["response_format"] = response_format
    return client.chat.completions.create(**options)
