"""Centralized LLM adapter for schema validation, retries, caching, and usage."""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Generic, TypeVar

from litellm.exceptions import (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
)
from litellm.exceptions import (
    APIError as LiteLLMAPIError,
)
from openai import (
    APIConnectionError as OpenAIConnectionError,
)
from openai import (
    APIError as OpenAIAPIError,
)
from openai import (
    APITimeoutError as OpenAITimeoutError,
)
from openai import (
    InternalServerError as OpenAIInternalServerError,
)
from openai import (
    RateLimitError as OpenAIRateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from article_rectifier.config import Settings
from article_rectifier.llm.openai_transport import openai_completion

SchemaT = TypeVar("SchemaT", bound=BaseModel)
CompletionFunction = Callable[..., object]
RETRYABLE_ERRORS = (
    APIConnectionError,
    InternalServerError,
    RateLimitError,
    ServiceUnavailableError,
    Timeout,
    OpenAIConnectionError,
    OpenAITimeoutError,
    OpenAIInternalServerError,
    OpenAIRateLimitError,
    ConnectionError,
    TimeoutError,
)


class LLMResponseStatus(StrEnum):
    """Describe the outcome of one validated logical LLM request."""

    SUCCESS = "success"
    CACHE_HIT = "cache_hit"
    INVALID_RESPONSE = "invalid_response"
    EMPTY_RESPONSE = "empty_response"
    TRANSPORT_FAILURE = "transport_failure"
    BUDGET_BLOCKED = "budget_blocked"
    OFFLINE_CACHE_MISS = "offline_cache_miss"


class LLMCallRecord(BaseModel):
    """Store non-secret metadata for one logical LLM request."""

    model_config = ConfigDict(frozen=True)

    prompt_version: str
    schema_name: str
    schema_version: str
    input_fingerprint: str
    model_identifier: str
    status: LLMResponseStatus
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(ge=0.0)
    retry_count: int = Field(default=0, ge=0)
    transport_calls: int | None = Field(default=None, ge=0)
    provider_cost_usd: float | None = Field(default=None, ge=0.0)
    error_code: str | None = None


@dataclass(frozen=True, slots=True)
class StructuredLLMResult(Generic[SchemaT]):
    """Return either one validated model or an explicit typed failure."""

    value: SchemaT | None
    record: LLMCallRecord

    @property
    def succeeded(self) -> bool:
        """Return whether a validated value is available.

        Returns:
            True for successful live and cached responses.
        """
        return self.value is not None


@dataclass(frozen=True, slots=True)
class _RequestContext(Generic[SchemaT]):
    """Bundle stable metadata shared by request outcome records."""

    schema: type[SchemaT]
    prompt_version: str
    schema_version: str
    cache_key: str
    started: float


def _read_field(value: object, name: str, default: object = None) -> object:
    """Read one field from a dynamic LiteLLM object or mapping.

    Args:
        value: LiteLLM response object or nested mapping.
        name: Field name.
        default: Value returned when the field is absent.

    Returns:
        The located field or default value.
    """
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _extract_content(response: object) -> str:
    """Extract exact message content from a LiteLLM response.

    Args:
        response: Dynamic LiteLLM completion response.

    Returns:
        Response content, or an empty string when absent.
    """
    choices = _read_field(response, "choices", [])
    if not isinstance(choices, (list, tuple)) or not choices:
        return ""
    message = _read_field(choices[0], "message", {})
    content = _read_field(message, "content", "")
    return content if isinstance(content, str) else ""


def _extract_usage(response: object) -> tuple[int, int, float | None]:
    """Extract observed tokens and optional provider cost.

    Args:
        response: Dynamic LiteLLM completion response.

    Returns:
        Input tokens, output tokens, and provider-reported cost when present.
    """
    usage = _read_field(response, "usage", {})
    input_tokens = _read_field(usage, "prompt_tokens", 0)
    output_tokens = _read_field(usage, "completion_tokens", 0)
    hidden = _read_field(response, "_hidden_params", {})
    cost = _read_field(hidden, "response_cost", None)
    return (
        int(input_tokens or 0),
        int(output_tokens or 0),
        float(cost) if cost else None,
    )


class LLMClient:
    """Make bounded proxy requests and validate every structured response."""

    def __init__(
        self,
        settings: Settings,
        completion_function: CompletionFunction = openai_completion,
        sleep_function: Callable[[float], None] = time.sleep,
    ) -> None:
        """Initialize the adapter without making a network request.

        Args:
            settings: Validated runtime settings.
            completion_function: Injectable LiteLLM-compatible transport.
            sleep_function: Injectable retry delay function.
        """
        self.settings = settings
        self._completion = completion_function
        self._sleep = sleep_function
        self._provider_cost_usd = 0.0
        self.trace_context = {
            "run_id": uuid.uuid4().hex,
            "article_id": "unassigned",
            "batch_id": "unassigned",
        }
        self._attempt_responses: list[object] = []
        self._attempt_count = 0
        self._request_metadata: dict[str, str] = {}

    def _trace_attempt(self, stage: str, **details: Any) -> None:
        """Persist per-attempt usage; include response diagnostics only in debug mode.

        Args:
            stage: Transport or parsing event name.
            **details: Non-secret request metadata and observations.
        """
        from article_rectifier.observability.debug import DebugTrace

        DebugTrace(self.settings.audit_directory / "calls").record(
            **self.trace_context,
            stage=stage,
            details={**self._request_metadata, **details},
        )

    def _cache_key(
        self,
        messages: list[dict[str, str]],
        prompt_version: str,
        schema: type[BaseModel],
        schema_version: str,
    ) -> str:
        """Build a stable cache key from all behavior-affecting inputs."""
        payload = {
            "messages": messages,
            "model": self.settings.llm_model_name,
            "prompt_version": prompt_version,
            "schema": schema.model_json_schema(),
            "schema_version": schema_version,
            "temperature": self.settings.temperature,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def _cache_path(self, cache_key: str) -> Path:
        """Resolve one response-cache path without creating it."""
        return self.settings.cache_directory / f"{cache_key}.json"

    def _load_cache(self, cache_key: str, schema: type[SchemaT]) -> SchemaT | None:
        """Load and revalidate a cached structured response."""
        if not self.settings.enable_llm_cache:
            return None
        path = self._cache_path(cache_key)
        if not path.is_file():
            return None
        try:
            return schema.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValidationError, ValueError):
            return None

    def _save_cache(self, cache_key: str, value: BaseModel) -> None:
        """Atomically store one validated response when caching is enabled."""
        if not self.settings.enable_llm_cache:
            return
        path = self._cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(value.model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    def _request_kwargs(self, messages: list[dict[str, str]]) -> dict[str, object]:
        """Build proxy transport arguments without logging the API key."""
        values: dict[str, object] = {
            "model": self.settings.llm_model_name,
            "messages": messages,
            "api_key": self.settings.require_api_key().get_secret_value(),
            "api_base": self.settings.llm_api_base,
            "temperature": self.settings.temperature,
            "timeout": self.settings.llm_timeout_seconds,
        }
        if self.settings.use_native_structured_output:
            values["response_format"] = {"type": "json_object"}
        return values

    def _call_transport(
        self,
        messages: list[dict[str, str]],
    ) -> tuple[object | None, int, str | None]:
        """Call the transport with bounded exponential backoff."""
        for retry in range(self.settings.max_llm_retries + 1):
            started = time.monotonic()
            self._attempt_count += 1
            attempt_id = uuid.uuid4().hex
            self._trace_attempt("llm_call_started", attempt_id=attempt_id, retry=retry)
            try:
                response = self._completion(**self._request_kwargs(messages))
                self._attempt_responses.append(response)
                inputs, outputs, cost = _extract_usage(response)
                self._trace_attempt(
                    "llm_call_finished",
                    attempt_id=attempt_id,
                    latency_seconds=time.monotonic() - started,
                    input_tokens=inputs,
                    output_tokens=outputs,
                    provider_cost_usd=cost,
                    status="response_received",
                )
                return response, retry, None
            except RETRYABLE_ERRORS as error:
                self._trace_attempt(
                    "llm_call_failed",
                    attempt_id=attempt_id,
                    latency_seconds=time.monotonic() - started,
                    error_type=type(error).__name__,
                    retryable=True,
                )
                if retry == self.settings.max_llm_retries:
                    return None, retry, type(error).__name__
                self._sleep(min(2**retry, 8))
            except (OpenAIAPIError, LiteLLMAPIError) as error:
                self._trace_attempt(
                    "llm_call_failed",
                    attempt_id=attempt_id,
                    latency_seconds=time.monotonic() - started,
                    error_type=type(error).__name__,
                    retryable=False,
                )
                return None, retry, type(error).__name__
        return None, self.settings.max_llm_retries, "transport_failure"

    def _record(
        self,
        *,
        prompt_version: str,
        schema: type[BaseModel],
        schema_version: str,
        cache_key: str,
        status: LLMResponseStatus,
        started: float,
        retry_count: int = 0,
        response: object | None = None,
        error_code: str | None = None,
    ) -> LLMCallRecord:
        """Build a complete usage and status record."""
        usages = [_extract_usage(item) for item in self._attempt_responses]
        costs = [item[2] for item in usages if item[2] is not None]
        tokens = (
            sum(item[0] for item in usages),
            sum(item[1] for item in usages),
            sum(costs) if costs else None,
        )
        if tokens[2] is not None:
            self._provider_cost_usd += tokens[2]
        return LLMCallRecord(
            prompt_version=prompt_version,
            schema_name=schema.__name__,
            schema_version=schema_version,
            input_fingerprint=cache_key,
            model_identifier=self.settings.llm_model_name,
            status=status,
            input_tokens=tokens[0],
            output_tokens=tokens[1],
            latency_seconds=time.monotonic() - started,
            retry_count=retry_count,
            transport_calls=self._attempt_count,
            provider_cost_usd=tokens[2],
            error_code=error_code,
        )

    def complete_structured(
        self,
        messages: list[dict[str, str]],
        schema: type[SchemaT],
        *,
        prompt_version: str,
        schema_version: str = "1",
    ) -> StructuredLLMResult[SchemaT]:
        """Request JSON, validate it, and permit one focused schema repair."""
        started = time.monotonic()
        self._attempt_responses = []
        self._attempt_count = 0
        key = self._cache_key(messages, prompt_version, schema, schema_version)
        self._request_metadata = {
            "input_fingerprint": key,
            "prompt_version": prompt_version,
            "schema": schema.__name__,
            "model": self.settings.llm_model_name,
        }
        cached = self._load_cache(key, schema)
        if cached is not None:
            record = self._record(
                prompt_version=prompt_version,
                schema=schema,
                schema_version=schema_version,
                cache_key=key,
                status=LLMResponseStatus.CACHE_HIT,
                started=started,
            )
            return StructuredLLMResult(cached, record)
        if self.settings.rectifier_cache_only:
            return self._failure(
                LLMResponseStatus.OFFLINE_CACHE_MISS,
                "cache_miss_offline",
                schema,
                prompt_version,
                schema_version,
                key,
                started,
                0,
                None,
            )
        if self._provider_cost_usd >= self.settings.max_run_cost_usd:
            return self._failure(
                LLMResponseStatus.BUDGET_BLOCKED,
                "run_cost_limit",
                schema,
                prompt_version,
                schema_version,
                key,
                started,
                0,
                None,
            )
        return self._complete_uncached(
            messages, schema, prompt_version, schema_version, key, started
        )

    def _complete_uncached(
        self,
        messages: list[dict[str, str]],
        schema: type[SchemaT],
        prompt_version: str,
        schema_version: str,
        cache_key: str,
        started: float,
    ) -> StructuredLLMResult[SchemaT]:
        """Execute one request and at most one invalid-JSON repair."""
        current_messages = list(messages)
        total_retries = 0
        context = _RequestContext(
            schema, prompt_version, schema_version, cache_key, started
        )
        for repair in range(2):
            response, retries, error = self._call_transport(current_messages)
            total_retries += retries
            if response is None:
                return self._context_failure(
                    context,
                    LLMResponseStatus.TRANSPORT_FAILURE,
                    error,
                    total_retries,
                    None,
                )
            value, status, error_code, content = self._parse_response(response, schema)
            if self.settings.rectifier_debug and status is not None:
                from article_rectifier.observability.debug import DebugTrace

                DebugTrace(self.settings.audit_directory / "debug").record(
                    **self.trace_context,
                    stage="llm_parse_failure",
                    details={
                        "input_fingerprint": cache_key,
                        "prompt_version": prompt_version,
                        "schema": schema.__name__,
                        "status": status,
                        "error_code": error_code,
                        "raw_content": content,
                    },
                )
            if status is LLMResponseStatus.EMPTY_RESPONSE:
                return self._context_failure(
                    context,
                    status,
                    error_code,
                    total_retries,
                    response,
                )
            if status is LLMResponseStatus.INVALID_RESPONSE:
                if repair == 0:
                    total_retries += 1
                    current_messages = self._repair_messages(
                        current_messages, content, schema
                    )
                    continue
                return self._context_failure(
                    context,
                    LLMResponseStatus.INVALID_RESPONSE,
                    "schema_validation",
                    total_retries,
                    response,
                )
            if value is None:
                raise RuntimeError("Validated LLM response did not contain a value")
            return self._success(
                value,
                response,
                total_retries,
                context,
            )
        raise RuntimeError("Structured response repair loop ended unexpectedly")

    def _success(
        self,
        value: SchemaT,
        response: object,
        retry_count: int,
        context: _RequestContext[SchemaT],
    ) -> StructuredLLMResult[SchemaT]:
        """Cache one valid value and create its complete success record.

        Args:
            value: Pydantic-validated response value.
            response: Dynamic LiteLLM response with optional usage metadata.
            retry_count: Transport and schema-repair retry count.
            context: Stable request metadata.

        Returns:
            Successful structured result with usage provenance.
        """
        self._save_cache(context.cache_key, value)
        record = self._record(
            prompt_version=context.prompt_version,
            schema=context.schema,
            schema_version=context.schema_version,
            cache_key=context.cache_key,
            status=LLMResponseStatus.SUCCESS,
            started=context.started,
            retry_count=retry_count,
            response=response,
        )
        return StructuredLLMResult(value, record)

    def _context_failure(
        self,
        context: _RequestContext[SchemaT],
        status: LLMResponseStatus,
        error_code: str | None,
        retry_count: int,
        response: object | None,
    ) -> StructuredLLMResult[SchemaT]:
        """Create a failure from bundled stable request metadata.

        Args:
            context: Stable request metadata.
            status: Typed response failure status.
            error_code: Optional transport or validation code.
            retry_count: Transport and schema-repair retry count.
            response: Dynamic response with optional usage metadata.

        Returns:
            Explicit structured failure without a model value.
        """
        return self._failure(
            status,
            error_code,
            context.schema,
            context.prompt_version,
            context.schema_version,
            context.cache_key,
            context.started,
            retry_count,
            response,
        )

    def _parse_response(
        self,
        response: object,
        schema: type[SchemaT],
    ) -> tuple[SchemaT | None, LLMResponseStatus | None, str | None, str]:
        """Extract and validate one exact structured response body.

        Args:
            response: Dynamic LiteLLM response.
            schema: Required Pydantic response model.

        Returns:
            Value, optional failure status, optional error code, and raw content.
        """
        content = _extract_content(response)
        if not content.strip():
            return None, LLMResponseStatus.EMPTY_RESPONSE, "empty_content", content
        try:
            return schema.model_validate_json(content), None, None, content
        except (ValidationError, ValueError) as error:
            if self.settings.rectifier_debug:
                self._trace_attempt("llm_schema_rejected", error_details=str(error))
            return (
                None,
                LLMResponseStatus.INVALID_RESPONSE,
                "schema_validation",
                content,
            )

    def _repair_messages(
        self,
        messages: list[dict[str, str]],
        invalid_content: str,
        schema: type[BaseModel],
    ) -> list[dict[str, str]]:
        """Append a focused request to repair malformed JSON only."""
        instruction = (
            "The previous response failed schema validation. Return only one JSON "
            f"object matching this schema: {json.dumps(schema.model_json_schema())}"
        )
        try:
            schema.model_validate_json(invalid_content)
        except (ValidationError, ValueError) as error:
            instruction += f"\nValidation errors to fix: {error}"
        return [
            *messages,
            {"role": "assistant", "content": invalid_content},
            {"role": "user", "content": instruction},
        ]

    def _failure(
        self,
        status: LLMResponseStatus,
        error_code: str | None,
        schema: type[SchemaT],
        prompt_version: str,
        schema_version: str,
        cache_key: str,
        started: float,
        retry_count: int,
        response: object | None,
    ) -> StructuredLLMResult[SchemaT]:
        """Build an explicit typed failure without inventing a model value."""
        record = self._record(
            prompt_version=prompt_version,
            schema=schema,
            schema_version=schema_version,
            cache_key=cache_key,
            status=status,
            started=started,
            retry_count=retry_count,
            response=response,
            error_code=error_code,
        )
        return StructuredLLMResult(None, record)

    @property
    def provider_cost_usd(self) -> float:
        """Return cumulative provider-reported cost observed by this client."""
        return self._provider_cost_usd


def response_as_dict(response: object) -> dict[str, Any]:
    """Convert a response for opt-in debugging without assuming its concrete type.

    Args:
        response: LiteLLM response object.

    Returns:
        JSON-compatible response data when available.
    """
    if isinstance(response, Mapping):
        return dict(response)
    dump = getattr(response, "model_dump", None)
    return dump() if callable(dump) else {}
