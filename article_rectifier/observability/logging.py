"""Compact JSONL audit logging without external monitoring services."""

from __future__ import annotations

import json
import threading
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

AuditValue = str | int | float | bool | None


class AuditEvent(BaseModel):
    """Represent one non-secret observable pipeline decision or transition."""

    model_config = ConfigDict(frozen=True)

    timestamp: str
    run_id: str = Field(min_length=1)
    article_id: str = Field(min_length=1)
    batch_id: str | None = None
    node_name: str = Field(min_length=1)
    event_type: str = Field(min_length=1)
    status: str = Field(min_length=1)
    duration_seconds: float = Field(default=0.0, ge=0.0)
    error_code: str | None = None
    details: dict[str, AuditValue] = Field(default_factory=dict)


class AuditLogger:
    """Append validated compact events to one local JSONL audit file."""

    def __init__(self, destination: Path) -> None:
        """Initialize a logger without creating its output file.

        Args:
            destination: JSONL output path outside final article directories.
        """
        self.destination = destination
        self._lock = threading.Lock()

    def emit(
        self,
        *,
        run_id: str,
        article_id: str,
        node_name: str,
        event_type: str,
        status: str,
        batch_id: str | None = None,
        duration_seconds: float = 0.0,
        error_code: str | None = None,
        details: dict[str, AuditValue] | None = None,
    ) -> AuditEvent:
        """Validate and append one event.

        Args:
            run_id: Stable identity for one process run.
            article_id: Current article identity.
            node_name: Workflow node or article orchestrator.
            event_type: Stable event vocabulary value.
            status: Accurate outcome status.
            batch_id: Optional current batch identity.
            duration_seconds: Observed node duration.
            error_code: Optional stable failure code.
            details: Compact non-secret counts or identifiers.

        Returns:
            The exact event written to disk.
        """
        event = AuditEvent(
            timestamp=datetime.now(UTC).isoformat(),
            run_id=run_id,
            article_id=article_id,
            batch_id=batch_id,
            node_name=node_name,
            event_type=event_type,
            status=status,
            duration_seconds=duration_seconds,
            error_code=error_code,
            details=details or {},
        )
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event.model_dump(mode="json"), sort_keys=True)
        with self._lock, self.destination.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        return event
