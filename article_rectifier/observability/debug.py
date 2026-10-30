"""Opt-in structured trace of model decisions and deterministic rejection gates."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


class DebugTrace:
    """Write bounded local JSONL traces without logging credentials or prompts."""

    def __init__(self, destination: Path, max_text_chars: int = 1200) -> None:
        """Set the trace directory and per-string safety limit.

        Args:
            destination: Local directory for per-run trace files.
            max_text_chars: Maximum characters kept per logged string.
        """
        self.destination = destination
        self.max_text_chars = max_text_chars
        self._lock = threading.Lock()

    def _safe(self, value: Any) -> Any:
        """Convert typed results into bounded JSON-compatible values.

        Args:
            value: Typed node result, collection, or scalar.

        Returns:
            Serialized value with long strings truncated.
        """
        if isinstance(value, BaseModel):
            return self._safe(value.model_dump(mode="json"))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, str):
            return value[: self.max_text_chars]
        if isinstance(value, dict):
            return {
                str(key): (
                    "[redacted]"
                    if str(key).lower()
                    in {"api_key", "llm_api_key", "authorization", "password", "secret"}
                    else self._safe(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, (list, tuple)):
            return [self._safe(item) for item in value]
        return value

    def record(
        self,
        *,
        run_id: str,
        article_id: str,
        batch_id: str,
        stage: str,
        details: dict[str, Any],
    ) -> None:
        """Append a node decision to the current run's debug trace.

        Args:
            run_id: Unique process-run identifier.
            article_id: Source article identifier.
            batch_id: Current editable batch identifier.
            stage: Explicit graph stage or decision point.
            details: Whitelisted node inputs and outputs, never credentials.
        """
        event = {
            "timestamp": datetime.now(UTC).isoformat(),
            "run_id": run_id,
            "article_id": article_id,
            "batch_id": batch_id,
            "stage": stage,
            "details": self._safe(details),
        }
        path = self.destination / f"{run_id}.jsonl"
        with self._lock:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as output:
                output.write(json.dumps(event, ensure_ascii=False) + "\n")
