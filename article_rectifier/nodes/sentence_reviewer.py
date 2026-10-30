"""Bounded operational review of changed facts in their full sentence context.

This model judgment is a pipeline gate, not an independent factuality metric.
"""

import json

from pydantic import BaseModel, ConfigDict, Field, model_validator

from article_rectifier.editing.diff_guard import apply_resolved_edits
from article_rectifier.llm.client import LLMClient, StructuredLLMResult
from article_rectifier.models.article import SentenceSpan
from article_rectifier.models.edits import SurgicalEdit
from article_rectifier.models.evidence import EvidenceSpan
from article_rectifier.models.results import Verdict


class SentenceReview(BaseModel):
    """Record the verdict, cited evidence, and actionable retry explanation."""

    model_config = ConfigDict(extra="forbid")
    verdict: Verdict
    evidence_ids: tuple[str, ...]
    explanation: str = Field(min_length=1, max_length=1200)
    failed_patch_ids: tuple[str, ...] = ()

    @model_validator(mode="after")
    def check_failed_ids(self) -> "SentenceReview":
        """Require targeted failures so valid neighboring edits are retained."""
        if self.verdict is not Verdict.SUPPORTED and not self.failed_patch_ids:
            raise ValueError("Non-supported reviews must identify failed_patch_ids")
        if self.verdict is Verdict.SUPPORTED and self.failed_patch_ids:
            raise ValueError("Supported reviews cannot contain failed_patch_ids")
        return self


def review_sentence(
    client: LLMClient,
    sentence: SentenceSpan,
    edits: list[SurgicalEdit],
    evidence: list[EvidenceSpan],
) -> StructuredLLMResult[SentenceReview]:
    """Check edited factual relationships and grammatical patch boundaries.

    Args:
        client: Central tracked model adapter.
        sentence: Immutable sentence before edits.
        edits: Deterministically validated edits with original global offsets.
        evidence: Exact relevant passages, including counter-evidence.

    Returns:
        Structured operational judgment with per-call provenance.
    """
    local = [
        edit.model_copy(
            update={
                "global_start": edit.global_start - sentence.start,
                "global_end": edit.global_end - sentence.start,
            }
        )
        for edit in edits
    ]
    payload = {
        "original": sentence.text,
        "candidate": apply_resolved_edits(sentence.text, local),
        "patches": [edit.model_dump(mode="json") for edit in edits],
        "evidence": [
            item.model_dump(mode="json", exclude={"retrieval_score"})
            for item in evidence
        ],
        "response_schema": SentenceReview.model_json_schema(),
    }
    instruction = (
        "Review proposed factual corrections using only the supplied source passages. "
        "Read the complete candidate, not just replacement words. Check the facts "
        "being edited, including negation, roles, comparisons and predecessor "
        "relationships. "
        "Use supported if those corrections are source-supported, minimal, and leave "
        "grammatical boundaries intact. Use contradicted if an edited fact remains "
        "wrong or introduces an incorrect relationship; explain the smallest needed "
        "repair. Use uncertain for conflicting or insufficient evidence. Do not demand "
        "style changes or changes to unrelated claims. A missing fact alone is not a "
        "contradiction. Cite supplied evidence IDs. Identify only the failed patch IDs "
        "in failed_patch_ids; use [] for supported. Check every patch independently "
        "within the complete sentence; do not reject a correct patch because another "
        "patch is wrong. Return JSON matching the schema."
    )
    return client.complete_structured(
        [
            {"role": "system", "content": instruction},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        SentenceReview,
        prompt_version="sentence-review-v2",
    )
