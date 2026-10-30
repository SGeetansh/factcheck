"""Offline scripted transport for deterministic LLM client tests."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable


class MockLLMTransport:
    """Return queued responses or raise queued exceptions without network access."""

    def __init__(self, outcomes: Iterable[object]) -> None:
        """Initialize a finite sequence of scripted transport outcomes.

        Args:
            outcomes: Response objects or exception instances in call order.
        """
        self._outcomes = deque(outcomes)
        self.calls: list[dict[str, object]] = []

    def __call__(self, **kwargs: object) -> object:
        """Record arguments and return the next scripted outcome.

        Args:
            **kwargs: LiteLLM-compatible request fields.

        Returns:
            The next queued response object.

        Raises:
            RuntimeError: If the queue is exhausted.
            BaseException: A queued exception outcome.
        """
        self.calls.append(kwargs)
        if not self._outcomes:
            raise RuntimeError("Mock LLM outcome queue is exhausted")
        outcome = self._outcomes.popleft()
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome
