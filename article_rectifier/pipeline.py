"""Sequential article-level orchestration over immutable sentence batches."""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from enum import StrEnum

from article_rectifier.config import Settings
from article_rectifier.editing.diff_guard import apply_resolved_edits
from article_rectifier.editing.resolution import unresolved_results
from article_rectifier.graph import BatchWorkflow
from article_rectifier.llm.client import LLMCallRecord, LLMClient
from article_rectifier.models.article import ArticleBatch, ArticleInput
from article_rectifier.models.edits import SurgicalEdit
from article_rectifier.models.results import (
    PatchResult,
    PatchStatus,
    VerificationResult,
)
from article_rectifier.models.state import BatchState
from article_rectifier.observability.logging import AuditLogger
from article_rectifier.preprocessing.batching import create_batches
from article_rectifier.preprocessing.sanitizer import (
    sanitize_generated_article,
    sanitize_source_article,
)
from article_rectifier.preprocessing.sentence_splitter import (
    reconstruct_from_spans,
    split_sentences,
)
from article_rectifier.retrieval.source_index import SourceIndex, build_source_index


class ArticleProcessingStatus(StrEnum):
    """Describe whether an article completed fully or with safe degradation."""

    COMPLETED = "completed"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class PipelineResult:
    """Return final article text and complete observable pipeline outcomes."""

    article_id: str
    output_text: str
    status: ArticleProcessingStatus
    accepted_edits: tuple[SurgicalEdit, ...]
    rejected_patches: tuple[PatchResult, ...]
    unresolved_issues: tuple[VerificationResult, ...]
    llm_records: tuple[LLMCallRecord, ...]
    errors: tuple[str, ...]
    source_hash: str
    original_hash: str


def _sha256(text: str) -> str:
    """Calculate a stable SHA-256 digest for immutable article text.

    Args:
        text: Unicode article content.

    Returns:
        Lowercase hexadecimal digest of UTF-8 bytes.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ArticleRectificationPipeline:
    """Process batches sequentially and merge validated original-offset patches."""

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize reusable batch workflow services.

        Args:
            settings: Validated runtime and safety configuration.
            llm_client: Centralized structured LLM adapter.
            audit_logger: Optional local JSONL event sink.
        """
        self.settings = settings
        self.audit_logger = audit_logger
        self.workflow = BatchWorkflow(settings, llm_client, audit_logger)

    def _initial_batch_state(
        self,
        article_id: str,
        run_id: str,
        source_text: str,
        original_text: str,
        source_index: SourceIndex,
        batch: ArticleBatch,
    ) -> BatchState:
        """Create complete state for one independent immutable batch.

        Args:
            article_id: Stable article identifier.
            run_id: Stable identity for the current process invocation.
            source_text: Sanitized authoritative source.
            original_text: Sanitized immutable generated article.
            source_index: Reusable per-article lexical source index.
            batch: Editable sentence batch and read-only context.

        Returns:
            Fully initialized typed graph state.
        """
        return {
            "article_id": article_id,
            "run_id": run_id,
            "batch": batch,
            "source_text": source_text,
            "source_index": source_index,
            "original_text": original_text,
            "fact_requests": [],
            "retrieved_evidence": {},
            "verification_results": [],
            "proposed_edits": [],
            "proposal_failures": {},
            "patch_results": [],
            "patch_history": [],
            "llm_records": [],
            "retry_count": 0,
            "status": "initialized",
            "candidate_text": original_text,
            "errors": [],
        }

    def rectify(self, article: ArticleInput) -> PipelineResult:
        """Rectify one article while retaining a safe unchanged fallback.

        Args:
            article: Immutable source and generated article input.

        Returns:
            Final article and all patch, issue, usage, and status observations.
        """
        source = sanitize_source_article(article.source_text)
        original = sanitize_generated_article(article.generated_text)
        run_id = uuid.uuid4().hex
        self._emit_article(run_id, article.article_id, "article_started", "started")
        sentences = split_sentences(original)
        reconstruct_from_spans(original, sentences)
        batches = create_batches(sentences, self.settings.batch_size)
        source_index = build_source_index(source, article.article_id)
        states: list[BatchState] = []
        for batch in batches:
            self._emit_article(
                run_id,
                article.article_id,
                "batch_created",
                "created",
                batch_id=batch.batch_id,
                sentence_count=len(batch.sentences),
            )
            states.append(
                self.workflow.invoke(
                    self._initial_batch_state(
                        article.article_id,
                        run_id,
                        source,
                        original,
                        source_index,
                        batch,
                    )
                )
            )
        result = self._merge(article.article_id, source, original, states)
        event = (
            "article_completed"
            if result.status is ArticleProcessingStatus.COMPLETED
            else "article_degraded"
        )
        self._emit_article(
            run_id,
            article.article_id,
            event,
            result.status.value,
            accepted_patches=len(result.accepted_edits),
            rejected_patches=len(result.rejected_patches),
        )
        return result

    def _emit_article(
        self,
        run_id: str,
        article_id: str,
        event_type: str,
        status: str,
        **details: str | int | float | bool | None,
    ) -> None:
        """Emit one article-level event when auditing is enabled.

        Args:
            run_id: Current process run identity.
            article_id: Stable article identity.
            event_type: Stable article lifecycle event.
            status: Accurate article status.
            **details: Compact non-secret result counts.
        """
        if self.audit_logger is not None:
            batch_id = details.pop("batch_id", None)
            self.audit_logger.emit(
                run_id=run_id,
                article_id=article_id,
                batch_id=str(batch_id) if batch_id is not None else None,
                node_name="article_orchestrator",
                event_type=event_type,
                status=status,
                details=details,
            )

    def _merge(
        self,
        article_id: str,
        source: str,
        original: str,
        states: list[BatchState],
    ) -> PipelineResult:
        """Merge final accepted batch patches against one immutable original."""
        results = [item for state in states for item in state["patch_results"]]
        accepted = [
            item.edit for item in results if item.status is PatchStatus.ACCEPTED
        ]
        rejected = [item for item in results if item.status is not PatchStatus.ACCEPTED]
        unresolved = [item for state in states for item in unresolved_results(state)]
        records = [item for state in states for item in state["llm_records"]]
        errors = [error for state in states for error in state["errors"]]
        output = apply_resolved_edits(original, accepted)
        degraded = bool(rejected or unresolved or errors)
        return PipelineResult(
            article_id=article_id,
            output_text=output,
            status=(
                ArticleProcessingStatus.DEGRADED
                if degraded
                else ArticleProcessingStatus.COMPLETED
            ),
            accepted_edits=tuple(accepted),
            rejected_patches=tuple(rejected),
            unresolved_issues=tuple(unresolved),
            llm_records=tuple(records),
            errors=tuple(errors),
            source_hash=_sha256(source),
            original_hash=_sha256(original),
        )
