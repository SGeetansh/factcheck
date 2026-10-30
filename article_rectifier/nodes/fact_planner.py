"""Structured fact-check planning with deterministic omission guards."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from article_rectifier.llm.client import LLMCallRecord, LLMClient
from article_rectifier.llm.prompts import PLANNER_PROMPT_VERSION, PLANNER_SYSTEM_PROMPT
from article_rectifier.models.article import ArticleBatch, SentenceSpan
from article_rectifier.models.claims import ErrorType, FactCheckPlan, FactCheckRequest

MONTHS = (
    "January|February|March|April|May|June|July|August|September|October|"
    "November|December"
)
HIGH_RISK_PATTERN = re.compile(
    rf"\b(?:{MONTHS})\s+\d{{1,2}}(?:,\s*\d{{4}})?\b|"
    r"(?<!\w)[₹$]?\d[\d,.]*(?:%|\s*(?:crore|lakh|million|billion|km|years?))?",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class PlannerOutcome:
    """Return a validated plan, optional call record, and semantic errors."""

    plan: FactCheckPlan
    call_record: LLMCallRecord | None
    errors: tuple[str, ...] = ()
    raw_plan: FactCheckPlan | None = None


def _anchor_error_type(anchor: str) -> ErrorType:
    """Classify an obvious date or numeric anchor.

    Args:
        anchor: Exact high-risk text found in a sentence.

    Returns:
        Temporal, magnitude, or numerical classification.
    """
    if re.search(MONTHS, anchor, re.IGNORECASE):
        return ErrorType.TEMPORAL
    if "%" in anchor or re.search(r"crore|lakh|million|billion", anchor, re.I):
        return ErrorType.MAGNITUDE
    return ErrorType.NUMERICAL


def _anchor_query(sentence: str, anchor: str) -> str:
    """Remove a possibly incorrect value while retaining stable search terms.

    Args:
        sentence: Complete editable sentence.
        anchor: Exact number or date to omit.

    Returns:
        Lexical query containing surrounding stable terms.
    """
    query = sentence.replace(anchor, " ")
    return " ".join(re.findall(r"[^\W_]+", query, re.UNICODE)) or sentence


def deterministic_anchor_requests(batch: ArticleBatch) -> list[FactCheckRequest]:
    """Create fact checks for dates and quantities that a planner may omit.

    Args:
        batch: Three or fewer editable sentence spans.

    Returns:
        Stable requests in sentence and anchor order.
    """
    requests: list[FactCheckRequest] = []
    for sentence in batch.sentences:
        for index, match in enumerate(
            HIGH_RISK_PATTERN.finditer(sentence.text), start=1
        ):
            anchor = match.group(0)
            requests.append(
                FactCheckRequest(
                    request_id=f"{batch.batch_id}-{sentence.sentence_id}-anchor-{index}",
                    sentence_id=sentence.sentence_id,
                    claim_text=sentence.text,
                    verification_question=f"What does the source state about {anchor}?",
                    search_query=_anchor_query(sentence.text, anchor),
                    error_category_hints=(_anchor_error_type(anchor),),
                    sentence_start=sentence.start,
                    sentence_end=sentence.end,
                )
            )
    return requests


def _batch_payload(batch: ArticleBatch, article_id: str) -> dict[str, object]:
    """Serialize only permitted planner input fields.

    Args:
        batch: Editable sentences and read-only neighbors.
        article_id: Stable article identity.

    Returns:
        JSON-compatible prompt payload without annotations or source truth.
    """
    return {
        "article_id": article_id,
        "batch_id": batch.batch_id,
        "editable_sentences": [
            sentence.model_dump(mode="json") for sentence in batch.sentences
        ],
        "context_before": (
            batch.context_before.model_dump(mode="json")
            if batch.context_before
            else None
        ),
        "context_after": (
            batch.context_after.model_dump(mode="json") if batch.context_after else None
        ),
        "response_schema": FactCheckPlan.model_json_schema(),
    }


def _validate_model_plan(
    plan: FactCheckPlan,
    batch: ArticleBatch,
) -> tuple[list[FactCheckRequest], list[str]]:
    """Reject requests that escape editable sentence boundaries.

    Args:
        plan: Schema-valid model plan.
        batch: Permitted editable batch.

    Returns:
        Semantically valid requests and rejection messages.
    """
    sentences = {sentence.sentence_id: sentence for sentence in batch.sentences}
    accepted: list[FactCheckRequest] = []
    errors: list[str] = []
    for request in plan.requests:
        sentence = sentences.get(request.sentence_id)
        valid = sentence is not None and _request_matches_sentence(request, sentence)
        if not valid:
            errors.append(f"invalid_request_boundary:{request.request_id}")
            continue
        # Models sometimes paraphrase a claim. Keep the verified sentence
        # identity and offsets, but use only verbatim article text downstream.
        if request.claim_text not in sentence.text:
            request = request.model_copy(update={"claim_text": sentence.text})
        accepted.append(request)
    return accepted, errors


def _request_matches_sentence(
    request: FactCheckRequest,
    sentence: SentenceSpan,
) -> bool:
    """Check one planner request against immutable sentence metadata.

    Args:
        request: Planner output request.
        sentence: Authoritative editable sentence span.

    Returns:
        True when identifiers, offsets, and claim text are permitted.
    """
    return (
        request.sentence_start == sentence.start
        and request.sentence_end == sentence.end
    )


def _merge_requests(
    model_requests: list[FactCheckRequest],
    anchors: list[FactCheckRequest],
) -> FactCheckPlan:
    """Merge model and deterministic requests without duplicate propositions.

    Args:
        model_requests: Semantically valid model-planned requests.
        anchors: Deterministic high-risk requests.

    Returns:
        One stable fact-check plan.
    """
    merged = list(model_requests)
    for anchor in anchors:
        covered = any(
            request.sentence_id == anchor.sentence_id
            and any(
                hint in request.error_category_hints
                for hint in anchor.error_category_hints
            )
            and request.claim_text == anchor.claim_text
            for request in model_requests
        )
        if not covered:
            merged.append(anchor)
    return FactCheckPlan(requests=tuple(merged))


class FactPlanner:
    """Identify verifiable claims without deciding truth or proposing edits."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize the planner with the centralized model adapter.

        Args:
            llm_client: Validated and observable LLM adapter.
        """
        self._llm = llm_client

    def plan(self, article_id: str, batch: ArticleBatch) -> PlannerOutcome:
        """Plan checks and add deterministic high-risk anchors.

        Args:
            article_id: Stable article identifier.
            batch: Editable sentences and read-only context.

        Returns:
            Validated plan with model-call provenance and semantic errors.
        """
        payload = _batch_payload(batch, article_id)
        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ]
        result = self._llm.complete_structured(
            messages,
            FactCheckPlan,
            prompt_version=PLANNER_PROMPT_VERSION,
        )
        errors: list[str] = []
        model_requests: list[FactCheckRequest] = []
        if result.value is None:
            errors.append(f"planner_llm:{result.record.status}")
        else:
            model_requests, errors = _validate_model_plan(result.value, batch)
        plan = _merge_requests(model_requests, deterministic_anchor_requests(batch))
        return PlannerOutcome(
            plan=plan,
            call_record=result.record,
            errors=tuple(errors),
            raw_plan=result.value,
        )
