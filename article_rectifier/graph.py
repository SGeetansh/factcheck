"""Explicit bounded LangGraph workflow for one immutable article batch."""

from __future__ import annotations

import time
from typing import Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from article_rectifier.config import Settings
from article_rectifier.editing.surgical_editor import execute_edits
from article_rectifier.llm.client import LLMClient
from article_rectifier.models.article import SentenceSpan
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import PatchStatus
from article_rectifier.models.state import BatchState
from article_rectifier.nodes.evidence_verifier import EvidenceVerifier
from article_rectifier.nodes.fact_planner import FactPlanner
from article_rectifier.nodes.patch_proposer import PatchProposer
from article_rectifier.nodes.validator import validate_patches
from article_rectifier.observability.logging import AuditLogger
from article_rectifier.retrieval.retriever import search_source


def _all_evidence(state: BatchState) -> list[EvidenceSpan]:
    """Flatten request evidence while preserving stable first occurrence.

    Args:
        state: Current batch workflow state.

    Returns:
        De-duplicated evidence in request and rank order.
    """
    unique: dict[str, EvidenceSpan] = {}
    for passages in state["retrieved_evidence"].values():
        for passage in passages:
            unique.setdefault(passage.evidence_id, passage)
    return list(unique.values())


def _sentence_map(state: BatchState) -> dict[str, SentenceSpan]:
    """Map editable sentence identities to immutable spans.

    Args:
        state: Current batch workflow state.

    Returns:
        Sentence ID mapping for the current editable batch.
    """
    return {item.sentence_id: item for item in state["batch"].sentences}


class BatchWorkflow:
    """Coordinate explicit batch nodes with one bounded patch retry."""

    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        """Initialize node services and compile a reusable StateGraph.

        Args:
            settings: Validated runtime and safety configuration.
            llm_client: Centralized structured LLM adapter.
            audit_logger: Optional local structured event sink.
        """
        self.settings = settings
        self.planner = FactPlanner(llm_client)
        self.verifier = EvidenceVerifier(llm_client)
        self.proposer = PatchProposer(llm_client)
        self.audit_logger = audit_logger
        self.compiled = self._build_graph()

    def _emit(
        self,
        state: BatchState,
        node_name: str,
        event_type: str,
        status: str,
        started: float,
        **details: str | int | float | bool | None,
    ) -> None:
        """Emit one node event when local auditing is configured.

        Args:
            state: Current typed batch state.
            node_name: Explicit graph node name.
            event_type: Stable decision or transition event.
            status: Accurate outcome status.
            started: Monotonic node start time.
            **details: Compact non-secret event observations.
        """
        if self.audit_logger is not None:
            self.audit_logger.emit(
                run_id=state["run_id"],
                article_id=state["article_id"],
                batch_id=state["batch"].batch_id,
                node_name=node_name,
                event_type=event_type,
                status=status,
                duration_seconds=time.monotonic() - started,
                details=details,
            )

    def _build_graph(self) -> CompiledStateGraph:
        """Compile explicit nodes and conditional retry transitions.

        Returns:
            Reusable compiled graph for typed batch state.
        """
        graph = StateGraph(BatchState)
        graph.add_node("planner", self._plan)
        graph.add_node("retrieve_evidence", self._retrieve)
        graph.add_node("verify_evidence", self._verify)
        graph.add_node("propose_patches", self._propose)
        graph.add_node("execute_patches", self._execute)
        graph.add_node("validate", self._validate)
        graph.add_node("prepare_retry", self._prepare_retry)
        graph.add_edge(START, "planner")
        graph.add_edge("planner", "retrieve_evidence")
        graph.add_edge("retrieve_evidence", "verify_evidence")
        graph.add_edge("verify_evidence", "propose_patches")
        graph.add_edge("propose_patches", "execute_patches")
        graph.add_edge("execute_patches", "validate")
        graph.add_conditional_edges(
            "validate",
            self._route_validation,
            {"retry": "prepare_retry", "end": END},
        )
        graph.add_edge("prepare_retry", "propose_patches")
        return graph.compile()

    def _plan(self, state: BatchState) -> dict[str, object]:
        """Run structured fact planning for the editable batch."""
        started = time.monotonic()
        outcome = self.planner.plan(state["article_id"], state["batch"])
        records = list(state["llm_records"])
        if outcome.call_record:
            records.append(outcome.call_record)
        self._emit(
            state,
            "planner",
            "facts_planned",
            "degraded" if outcome.errors else "completed",
            started,
            fact_count=len(outcome.plan.requests),
        )
        return {
            "fact_requests": list(outcome.plan.requests),
            "llm_records": records,
            "errors": [*state["errors"], *outcome.errors],
        }

    def _retrieve(self, state: BatchState) -> dict[str, object]:
        """Retrieve exact source evidence for each planned request."""
        started = time.monotonic()
        retrieved = {
            request.request_id: search_source(
                request.search_query,
                state["source_index"],
                self.settings.retrieval_top_k,
                source_anchors=(request.claim_text,),
            )
            for request in state["fact_requests"]
        }
        self._emit(
            state,
            "retrieve_evidence",
            "evidence_retrieved",
            "completed",
            started,
            evidence_count=sum(len(items) for items in retrieved.values()),
            retrieval_failures=sum(not items for items in retrieved.values()),
        )
        return {"retrieved_evidence": retrieved}

    def _verify(self, state: BatchState) -> dict[str, object]:
        """Verify every request against its exact retrieved passages."""
        started = time.monotonic()
        verifications = []
        records = list(state["llm_records"])
        errors = list(state["errors"])
        for request in state["fact_requests"]:
            outcome = self.verifier.verify(
                request,
                state["retrieved_evidence"].get(request.request_id, []),
                state["source_text"],
            )
            verifications.append(outcome.result)
            if outcome.call_record:
                records.append(outcome.call_record)
            errors.extend(outcome.errors)
        self._emit(
            state,
            "verify_evidence",
            "claim_verified",
            "degraded" if len(errors) > len(state["errors"]) else "completed",
            started,
            verification_count=len(verifications),
        )
        return {
            "verification_results": verifications,
            "llm_records": records,
            "errors": errors,
        }

    def _propose(self, state: BatchState) -> dict[str, object]:
        """Propose minimal edits for all actionable contradictions."""
        started = time.monotonic()
        requests = {item.request_id: item for item in state["fact_requests"]}
        sentences = _sentence_map(state)
        proposed = []
        records = list(state["llm_records"])
        errors = list(state["errors"])
        feedback = (
            "; ".join(
                item.reason or "validation failed"
                for item in state["patch_results"]
                if item.status is PatchStatus.RETRYABLE
            )
            or None
        )
        for verification in state["verification_results"]:
            request = requests[verification.request_id]
            outcome = self.proposer.propose(
                sentences[request.sentence_id],
                request,
                verification,
                state["retrieved_evidence"].get(request.request_id, []),
                retry_feedback=feedback,
                retry_count=state["retry_count"],
            )
            proposed.extend(outcome.proposal.edits)
            if outcome.call_record:
                records.append(outcome.call_record)
            errors.extend(outcome.errors)
        self._emit(
            state,
            "propose_patches",
            "patch_proposed",
            "degraded" if len(errors) > len(state["errors"]) else "completed",
            started,
            patch_count=len(proposed),
        )
        return {"proposed_edits": proposed, "llm_records": records, "errors": errors}

    def _execute(self, state: BatchState) -> dict[str, object]:
        """Resolve and execute only authorized non-conflicting edits."""
        started = time.monotonic()
        execution = execute_edits(
            state["original_text"],
            state["proposed_edits"],
            list(state["batch"].sentences),
            state["verification_results"],
            _all_evidence(state),
            state["source_text"],
        )
        accepted = sum(
            item.status is PatchStatus.ACCEPTED for item in execution.results
        )
        event_type = (
            "patch_accepted"
            if accepted
            else "patch_rejected" if execution.results else "patch_skipped"
        )
        self._emit(
            state,
            "execute_patches",
            event_type,
            "completed",
            started,
            accepted_patches=accepted,
            rejected_patches=len(execution.results) - accepted,
        )
        return {"patch_results": list(execution.results), "status": execution.text}

    def _validate(self, state: BatchState) -> dict[str, object]:
        """Validate provisional output and roll back every unsafe patch."""
        started = time.monotonic()
        validation = validate_patches(
            state["original_text"],
            state["status"],
            state["patch_results"],
            list(state["batch"].sentences),
            _all_evidence(state),
            state["source_text"],
            retry_count=state["retry_count"],
            max_patch_retries=self.settings.max_patch_retries,
            soft_patch_character_limit=self.settings.soft_patch_character_limit,
        )
        status = (
            "retryable"
            if any(item.status is PatchStatus.RETRYABLE for item in validation.results)
            else "completed"
        )
        errors = [*state["errors"], *validation.warnings]
        self._emit(
            state,
            "validate",
            "retry_requested" if status == "retryable" else "batch_completed",
            status,
            started,
            accepted_patches=sum(
                item.status is PatchStatus.ACCEPTED for item in validation.results
            ),
        )
        return {
            "patch_results": list(validation.results),
            "status": status,
            "errors": errors,
        }

    def _route_validation(self, state: BatchState) -> Literal["retry", "end"]:
        """Route one bounded targeted retry or finish the batch."""
        if (
            state["status"] == "retryable"
            and state["retry_count"] < self.settings.max_patch_retries
        ):
            return "retry"
        return "end"

    def _prepare_retry(self, state: BatchState) -> dict[str, object]:
        """Preserve failed dispositions and increment the retry counter."""
        return {
            "patch_history": [*state["patch_history"], *state["patch_results"]],
            "retry_count": state["retry_count"] + 1,
        }

    def invoke(self, state: BatchState) -> BatchState:
        """Run the compiled bounded workflow for one batch.

        Args:
            state: Fully initialized batch state.

        Returns:
            Final typed batch state.
        """
        return cast(BatchState, self.compiled.invoke(state))
