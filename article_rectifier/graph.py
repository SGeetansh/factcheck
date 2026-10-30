"""Explicit bounded LangGraph workflow for one immutable article batch."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any, Literal, cast

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from article_rectifier.config import Settings
from article_rectifier.editing.diff_guard import apply_resolved_edits
from article_rectifier.editing.minimal_edits import changed_fragments
from article_rectifier.editing.retry import merge_retry_results
from article_rectifier.editing.surgical_editor import execute_edits
from article_rectifier.llm.client import LLMCallRecord, LLMClient
from article_rectifier.models.article import SentenceSpan
from article_rectifier.models.evidence import EvidenceSpan, RetrievalStrategy
from article_rectifier.models.results import PatchStatus, Verdict, VerificationResult
from article_rectifier.models.state import BatchState
from article_rectifier.nodes.batch_verifier import verify_batch
from article_rectifier.nodes.evidence_verifier import EvidenceVerifier
from article_rectifier.nodes.fact_planner import FactPlanner
from article_rectifier.nodes.patch_proposer import PatchProposer, _is_actionable
from article_rectifier.nodes.sentence_reviewer import review_sentence
from article_rectifier.nodes.validator import ValidationExecution, validate_patches
from article_rectifier.observability.debug import DebugTrace
from article_rectifier.observability.logging import AuditLogger
from article_rectifier.retrieval.retriever import search_source

LOGGER = logging.getLogger(__name__)


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
        self.llm_client = llm_client
        self.planner = FactPlanner(llm_client)
        self.verifier = EvidenceVerifier(llm_client)
        self.proposer = PatchProposer(llm_client)
        self.audit_logger = audit_logger
        self.debug_trace = (
            DebugTrace(settings.audit_directory / "debug")
            if settings.rectifier_debug
            else None
        )
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

    def _trace(self, state: BatchState, stage: str, **details: Any) -> None:
        """Record stage input/output decisions when debug tracing is enabled.

        Args:
            state: Batch state containing stable trace identifiers.
            stage: Name of the traced graph transition.
            **details: Typed output fields and rejection explanations.
        """
        if self.debug_trace is None:
            return
        self.debug_trace.record(
            run_id=state["run_id"],
            article_id=state["article_id"],
            batch_id=state["batch"].batch_id,
            stage=stage,
            details=details,
        )

    def _build_graph(self) -> CompiledStateGraph:
        """Compile explicit nodes and conditional retry transitions.

        Returns:
            Reusable compiled graph for typed batch state.
        """
        graph = StateGraph(BatchState)
        for name, node in (
            ("planner", self._plan),
            ("retrieve_evidence", self._retrieve),
            ("verify_evidence", self._verify),
            ("propose_patches", self._propose),
            ("execute_patches", self._execute),
            ("validate", self._validate),
            ("prepare_retry", self._prepare_retry),
        ):
            graph.add_node(name, self._timed_node(name, node))
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

    def _timed_node(
        self, name: str, node: Callable[[BatchState], dict[str, object]]
    ) -> Callable[[BatchState], dict[str, object]]:
        """Wrap a node with timed, correlated input/output and failure traces.

        Args:
            name: Stable graph node name.
            node: State transformation to observe.

        Returns:
            Instrumented transformation with identical return values.
        """

        def observed(state: BatchState) -> dict[str, object]:
            started = time.monotonic()
            self.llm_client.trace_context = {
                "run_id": state["run_id"],
                "article_id": state["article_id"],
                "batch_id": state["batch"].batch_id,
            }
            snapshot = {
                key: value
                for key, value in state.items()
                if key not in {"source_index", "source_text", "original_text"}
            }
            self._trace(state, "node_started", node=name, state_snapshot=snapshot)
            LOGGER.info(
                "%s/%s %s started", state["article_id"], state["batch"].batch_id, name
            )
            try:
                update = node(state)
            except Exception as error:
                self._trace(
                    state,
                    "node_failed",
                    node=name,
                    duration_seconds=time.monotonic() - started,
                    error_type=type(error).__name__,
                )
                raise
            elapsed = time.monotonic() - started
            self._trace(
                state,
                "node_finished",
                node=name,
                duration_seconds=elapsed,
                update=update,
            )
            LOGGER.info(
                "%s/%s %s finished in %.3fs",
                state["article_id"],
                state["batch"].batch_id,
                name,
                elapsed,
            )
            return update

        return observed

    def _plan(self, state: BatchState) -> dict[str, object]:
        """Run structured fact planning for the editable batch."""
        started = time.monotonic()
        outcome = self.planner.plan(state["article_id"], state["batch"])
        self._trace(
            state,
            "planner_output",
            editable_sentences=state["batch"].sentences,
            raw_plan=outcome.raw_plan,
            accepted_requests=outcome.plan.requests,
            rejected_codes=outcome.errors,
            llm_record=outcome.call_record,
        )
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
            rejection_codes=";".join(outcome.errors)[:500],
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
        for request in state["fact_requests"]:
            self._trace(
                state,
                "retrieval_output",
                request_id=request.request_id,
                question=request.verification_question,
                query=request.search_query,
                evidence=retrieved.get(request.request_id, []),
            )
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
        if self.settings.batch_verification:
            return self._verify_together(state)
        started = time.monotonic()
        verifications = []
        records = list(state["llm_records"])
        errors = list(state["errors"])
        retrieved = dict(state["retrieved_evidence"])
        for request in state["fact_requests"]:
            outcome = self.verifier.verify(
                request,
                state["retrieved_evidence"].get(request.request_id, []),
                state["source_text"],
            )
            if (
                outcome.result.verdict
                in {Verdict.UNSUPPORTED, Verdict.UNCERTAIN, Verdict.CONTRADICTED}
                and not outcome.errors
            ):
                anchor = (
                    request.claim_text
                    if outcome.result.verdict is Verdict.CONTRADICTED
                    else request.search_query
                )
                expanded = search_source(
                    f"{request.verification_question} {anchor}",
                    state["source_index"],
                    min(12, self.settings.retrieval_top_k * 2),
                )
                existing = {
                    item.evidence_id: item for item in retrieved[request.request_id]
                }
                if any(item.evidence_id not in existing for item in expanded):
                    self._trace(
                        state,
                        "retrieval_retry",
                        request_id=request.request_id,
                        initial_result=outcome.result,
                        initial_call=outcome.call_record,
                        purpose=(
                            "seek_counterevidence"
                            if outcome.result.verdict is Verdict.CONTRADICTED
                            else "resolve_insufficient_evidence"
                        ),
                        expanded_evidence=expanded,
                    )
                    if outcome.call_record:
                        records.append(outcome.call_record)
                    existing.update({item.evidence_id: item for item in expanded})
                    retrieved[request.request_id] = list(existing.values())
                    outcome = self.verifier.verify(
                        request, list(existing.values()), state["source_text"]
                    )
            verifications.append(outcome.result)
            self._trace(
                state,
                "verifier_output",
                request_id=request.request_id,
                original_claim=request.claim_text,
                raw_result=outcome.raw_result,
                effective_result=outcome.result,
                rejected_codes=outcome.errors,
                llm_record=outcome.call_record,
            )
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
            actionable_count=sum(
                item.verdict.value == "contradicted"
                and item.status.value == "correction_available"
                for item in verifications
            ),
            rejection_codes=";".join(errors[len(state["errors"]) :])[:500],
        )
        return {
            "verification_results": verifications,
            "retrieved_evidence": retrieved,
            "llm_records": records,
            "errors": errors,
        }

    def _verify_together(self, state: BatchState) -> dict[str, object]:
        """Retrieve counter-evidence first and verify the batch in one model call.

        Args:
            state: Planned requests and deterministic primary retrieval results.

        Returns:
            Validated judgments, expanded evidence, errors, and usage records.
        """
        started = time.monotonic()
        retrieved = dict(state["retrieved_evidence"])
        for request in state["fact_requests"]:
            expanded = search_source(
                f"{request.verification_question} {request.claim_text}",
                state["source_index"],
                self.settings.retrieval_top_k,
            )
            passages = {
                item.evidence_id: item for item in retrieved[request.request_id]
            }
            passages.update({item.evidence_id: item for item in expanded})
            retrieved[request.request_id] = list(passages.values())
        batch = state["batch"]
        context = "\n".join(
            item.text
            for item in [batch.context_before, *batch.sentences, batch.context_after]
            if item is not None
        )
        outcome = verify_batch(
            self.llm_client,
            state["fact_requests"],
            retrieved,
            state["source_text"],
            article_context=context,
        )
        initial_record = outcome.record
        unknown = {
            item.request_id
            for item in outcome.results
            if item.verdict in {Verdict.UNSUPPORTED, Verdict.UNCERTAIN}
        }
        recovery_record = None
        if unknown:
            source = state["source_text"]
            # Bound the broader context and retain exact source offsets.
            end = (
                len(source) if len(source) <= 16000 else source.rfind("\n\n", 0, 16000)
            )
            end = end if end > 0 else min(16000, len(source))
            broad = EvidenceSpan(
                evidence_id=f"src-{state['article_id']}-context-{end}",
                source_article_id=state["article_id"],
                text=source[:end],
                start=0,
                end=end,
                retrieval_score=0,
                retrieval_strategy=RetrievalStrategy.SOURCE_CENTRIC,
            )
            for request_id in sorted(unknown):
                retrieved[request_id] = [*retrieved[request_id], broad]
            recovery = verify_batch(
                self.llm_client,
                [item for item in state["fact_requests"] if item.request_id in unknown],
                {key: retrieved[key] for key in sorted(unknown)},
                source,
                recovery=True,
                article_context=context,
            )
            recovery_record = recovery.record
            replacements = {item.request_id: item for item in recovery.results}
            self._trace(
                state,
                "verification_recovery",
                initial_results=outcome.results,
                recovery_results=recovery.results,
                llm_record=recovery.record,
            )
            outcome = type(outcome)(
                tuple(
                    replacements.get(item.request_id, item) for item in outcome.results
                ),
                outcome.record,
                (*outcome.errors, *recovery.errors),
            )
        shared = {
            item.evidence_id: item for spans in retrieved.values() for item in spans
        }
        for result in outcome.results:
            passages = {item.evidence_id: item for item in retrieved[result.request_id]}
            passages.update({key: shared[key] for key in result.evidence_ids})
            retrieved[result.request_id] = list(passages.values())
        self._trace(
            state,
            "batch_verification",
            requests=state["fact_requests"],
            evidence=retrieved,
            results=outcome.results,
            errors=outcome.errors,
            llm_record=outcome.record,
        )
        self._emit(
            state,
            "verify_evidence",
            "claim_verified",
            "degraded" if outcome.errors else "completed",
            started,
            verification_count=len(outcome.results),
            actionable_count=sum(_is_actionable(item) for item in outcome.results),
        )
        return {
            "verification_results": list(outcome.results),
            "retrieved_evidence": retrieved,
            "errors": [*state["errors"], *outcome.errors],
            "llm_records": [
                *state["llm_records"],
                *([initial_record] if initial_record else []),
                *([recovery_record] if recovery_record else []),
            ],
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
        groups: dict[str, list[VerificationResult]] = {}
        failures: dict[str, str] = {}
        retry_ids = {
            item.edit.request_id
            for item in state["patch_results"]
            if item.status is PatchStatus.RETRYABLE
        }
        retry_ids.update(state.get("proposal_failures", {}))
        for verification in state["verification_results"]:
            if _is_actionable(verification) and (
                not state["retry_count"] or verification.request_id in retry_ids
            ):
                sentence_id = requests[verification.request_id].sentence_id
                groups.setdefault(sentence_id, []).append(verification)
        for sentence_id, verifications in groups.items():
            group_requests = [requests[item.request_id] for item in verifications]
            evidence = {
                item.evidence_id: item
                for request in group_requests
                for item in state["retrieved_evidence"].get(request.request_id, [])
            }
            failed = [
                item
                for item in state["patch_results"]
                if item.status is PatchStatus.RETRYABLE
                and item.edit.sentence_id == sentence_id
            ]
            feedback = (
                json.dumps(
                    {
                        "failed_patches": [
                            {
                                "original_text": item.edit.original_text,
                                "replacement_text": item.edit.replacement_text,
                                "reason": item.reason,
                            }
                            for item in failed
                        ],
                        "proposal_failures": {
                            item.request_id: state.get("proposal_failures", {}).get(
                                item.request_id
                            )
                            for item in verifications
                        },
                        "instruction": "Retry ONLY failed patches. Keep accepted "
                        "patches unchanged. Use independent minimal replacements; "
                        "retain surrounding prose. If a target is ambiguous, include "
                        "unchanged surrounding words to identify each occurrence.",
                    }
                )
                if state["retry_count"]
                else None
            )
            outcome = self.proposer.propose_group(
                sentences[sentence_id],
                group_requests,
                verifications,
                list(evidence.values()),
                retry_feedback=feedback,
                retry_count=state["retry_count"],
            )
            safe = list(outcome.proposal.edits)
            if failed:
                safe = [
                    edit
                    for edit in safe
                    if all(
                        any(
                            edit.original_text[start:end] in item.edit.original_text
                            for item in failed
                        )
                        for start, end, _ in changed_fragments(
                            edit.original_text, edit.replacement_text
                        )
                    )
                ]
            if not safe or len(safe) != len(outcome.proposal.edits) or outcome.errors:
                reason = "; ".join(outcome.errors) or "missing_or_out_of_scope_patch"
                failures.update({item.request_id: reason for item in verifications})
            proposed.extend(
                edit.model_copy(
                    update={
                        "patch_id": f"{edit.request_id}:"
                        + (
                            f"retry{state['retry_count']}:"
                            if state["retry_count"]
                            else ""
                        )
                        + edit.patch_id
                    }
                )
                for edit in safe
            )
            self._trace(
                state,
                "proposer_output",
                request_ids=[item.request_id for item in verifications],
                sentence_id=sentence_id,
                retry_count=state["retry_count"],
                retry_feedback=feedback,
                raw_proposal=outcome.raw_proposal,
                accepted_proposals=outcome.proposal.edits,
                rejected_codes=outcome.errors,
                llm_record=outcome.call_record,
            )
            if outcome.call_record:
                records.append(outcome.call_record)
            errors.extend(outcome.errors)
        actionable = sum(
            result.verdict.value == "contradicted"
            and result.status.value == "correction_available"
            for result in state["verification_results"]
        )
        if actionable and not proposed:
            errors.append(f"actionable_without_patches:{actionable}")
        self._emit(
            state,
            "propose_patches",
            "patch_proposed",
            "degraded" if len(errors) > len(state["errors"]) else "completed",
            started,
            actionable_count=actionable,
            patch_count=len(proposed),
            rejection_codes=";".join(errors[len(state["errors"]) :])[:500],
        )
        return {
            "proposed_edits": proposed,
            "proposal_failures": failures,
            "llm_records": records,
            "errors": errors,
        }

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
        candidate, results = merge_retry_results(
            state["original_text"],
            state["patch_results"],
            list(execution.results),
            state["retry_count"],
            self.settings.max_patch_retries,
        )
        accepted = sum(item.status is PatchStatus.ACCEPTED for item in results)
        self._trace(
            state,
            "executor_output",
            proposed_edits=state["proposed_edits"],
            patch_results=results,
            changed=candidate != state["original_text"],
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
            rejected_patches=len(results) - accepted,
        )
        return {
            "patch_results": results,
            "candidate_text": candidate,
        }

    def _validate(self, state: BatchState) -> dict[str, object]:
        """Validate provisional output and roll back every unsafe patch."""
        started = time.monotonic()
        validation = validate_patches(
            state["original_text"],
            state["candidate_text"],
            state["patch_results"],
            list(state["batch"].sentences),
            _all_evidence(state),
            state["source_text"],
            retry_count=state["retry_count"],
            max_patch_retries=self.settings.max_patch_retries,
            soft_patch_character_limit=self.settings.soft_patch_character_limit,
            semantic_review_enabled=self.settings.enable_validation_llm,
        )
        records = list(state["llm_records"])
        if self.settings.enable_validation_llm:
            validation, review_records = self._review_changes(state, validation)
            records.extend(review_records)
        status = (
            "retryable"
            if any(item.status is PatchStatus.RETRYABLE for item in validation.results)
            or state.get("proposal_failures")
            else "completed"
        )
        errors = [*state["errors"], *validation.warnings]
        self._trace(
            state,
            "validator_output",
            before=state["patch_results"],
            after=validation.results,
            warnings=validation.warnings,
            candidate_changed=state["candidate_text"] != state["original_text"],
            final_changed=validation.text != state["original_text"],
            route=status,
        )
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
            "candidate_text": validation.text,
            "llm_records": records,
            "status": status,
            "errors": errors,
        }

    def _review_changes(
        self, state: BatchState, validation: ValidationExecution
    ) -> tuple[ValidationExecution, list[LLMCallRecord]]:
        """Review changed sentences and retain one bounded semantic repair.

        Args:
            state: Original inputs and verified source evidence.
            validation: Results after deterministic structural checks.

        Returns:
            Updated patch validation and tracked review-call records.
        """
        results = list(validation.results)
        records = []
        for sentence in state["batch"].sentences:
            # Review the complete candidate after pending factual repairs. Reviewing
            # a partially rolled-back sentence can invalidate an unrelated safe edit.
            if any(
                item.status is PatchStatus.RETRYABLE
                and item.edit.sentence_id == sentence.sentence_id
                for item in results
            ):
                continue
            edits = [
                item.edit
                for item in results
                if item.status is PatchStatus.ACCEPTED
                and item.edit.sentence_id == sentence.sentence_id
            ]
            if not edits:
                continue
            evidence = {
                item.evidence_id: item
                for request in state["fact_requests"]
                if request.sentence_id == sentence.sentence_id
                for item in state["retrieved_evidence"].get(request.request_id, [])
            }
            response = review_sentence(
                self.llm_client, sentence, edits, list(evidence.values())
            )
            records.append(response.record)
            review = response.value
            edit_ids = {edit.patch_id for edit in edits}
            valid_review = bool(
                review
                and review.evidence_ids
                and set(review.evidence_ids) <= set(evidence)
                and set(review.failed_patch_ids) <= edit_ids
            )
            accepted = bool(
                review and review.verdict is Verdict.SUPPORTED and valid_review
            )
            self._trace(
                state,
                "sentence_review",
                sentence_id=sentence.sentence_id,
                review=review,
                accepted=accepted,
                llm_record=response.record,
            )
            if accepted:
                continue
            status = (
                PatchStatus.RETRYABLE
                if state["retry_count"] < self.settings.max_patch_retries
                else PatchStatus.UNRESOLVED
            )
            reason = (
                review.explanation if review else f"review_llm:{response.record.status}"
            )
            failed_ids = set(review.failed_patch_ids) if valid_review else edit_ids
            results = [
                (
                    item.model_copy(update={"status": status, "reason": reason})
                    if item.edit in edits and item.edit.patch_id in failed_ids
                    else item
                )
                for item in results
            ]
        output = apply_resolved_edits(
            state["original_text"],
            [item.edit for item in results if item.status is PatchStatus.ACCEPTED],
        )
        return ValidationExecution(output, tuple(results), validation.warnings), records

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
        self._trace(
            state,
            "retry_transition",
            old_retry_count=state["retry_count"],
            prior_patch_results=state["patch_results"],
        )
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
        self._trace(
            state,
            "batch_started",
            editable_sentences=state["batch"].sentences,
            retry_limit=self.settings.max_patch_retries,
        )
        final_state = cast(BatchState, self.compiled.invoke(state))
        self._trace(
            final_state,
            "batch_finished",
            status=final_state["status"],
            patch_results=final_state["patch_results"],
            errors=final_state["errors"],
            retry_count=final_state["retry_count"],
            llm_records=final_state["llm_records"],
        )
        return final_state
