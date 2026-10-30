"""Offline tests for minimal evidence-backed patch proposals."""

import json
from pathlib import Path

import pytest

from article_rectifier.config import Settings
from article_rectifier.llm import LLMClient, MockLLMTransport
from article_rectifier.models import (
    EditOperation,
    ErrorType,
    EvidenceSpan,
    FactCheckRequest,
    RetrievalStrategy,
    SentenceSpan,
    Verdict,
    VerificationResult,
    VerificationStatus,
)
from article_rectifier.nodes.patch_proposer import PatchProposer

SENTENCE_TEXT = "The film was blocked in Karnataka on August 15, 2024."
EVIDENCE_TEXT = "The film opened September 5, 2025, and was blocked in West Bengal."


def sentence() -> SentenceSpan:
    """Create the only permitted editable sentence.

    Returns:
        Sentence fixture with immutable offsets.
    """
    return SentenceSpan(
        sentence_id="s-1",
        text=SENTENCE_TEXT,
        start=10,
        end=10 + len(SENTENCE_TEXT),
        paragraph_id="p-1",
    )


def request() -> FactCheckRequest:
    """Create a geographic fact-check request.

    Returns:
        Request fixture tied to the permitted sentence.
    """
    return FactCheckRequest(
        request_id="req-1",
        sentence_id="s-1",
        claim_text="blocked in Karnataka",
        verification_question="Where was the film blocked?",
        search_query="film blocked theaters state",
        error_category_hints=(ErrorType.GEOGRAPHIC,),
        sentence_start=10,
        sentence_end=10 + len(SENTENCE_TEXT),
    )


def evidence() -> EvidenceSpan:
    """Create one exact supporting source passage.

    Returns:
        Evidence fixture.
    """
    return EvidenceSpan(
        evidence_id="ev-1",
        source_article_id="article-4",
        text=EVIDENCE_TEXT,
        start=0,
        end=len(EVIDENCE_TEXT),
        retrieval_score=2.0,
        retrieval_strategy=RetrievalStrategy.BM25,
    )


def verification(
    *,
    verdict: Verdict = Verdict.CONTRADICTED,
    status: VerificationStatus = VerificationStatus.CORRECTION_AVAILABLE,
) -> VerificationResult:
    """Create an actionable or overridden verification result.

    Args:
        verdict: Source-support classification.
        status: Correction availability.

    Returns:
        Validated verification fixture.
    """
    correction = (
        "West Bengal" if status is VerificationStatus.CORRECTION_AVAILABLE else None
    )
    error_types = (ErrorType.GEOGRAPHIC,) if correction else ()
    evidence_ids = (
        ("ev-1",) if verdict in {Verdict.CONTRADICTED, Verdict.SUPPORTED} else ()
    )
    return VerificationResult(
        request_id="req-1",
        verdict=verdict,
        evidence_ids=evidence_ids,
        explanation="The source establishes the location.",
        error_types=error_types,
        proposed_factual_correction=correction,
        status=status,
    )


def make_proposer(
    tmp_path: Path,
    edits: list[dict[str, object]],
) -> tuple[PatchProposer, MockLLMTransport]:
    """Create a proposer with a scripted patch response.

    Args:
        tmp_path: Temporary project root.
        edits: JSON-compatible edit proposals.

    Returns:
        Proposer and inspectable mock transport.
    """
    content = json.dumps({"edits": edits})
    transport = MockLLMTransport([{"choices": [{"message": {"content": content}}]}])
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        max_llm_retries=0,
        _env_file=None,
    )
    return PatchProposer(LLMClient(settings, transport)), transport


def edit(**updates: object) -> dict[str, object]:
    """Create one minimal geographic replacement mapping.

    Args:
        **updates: Fields to override.

    Returns:
        JSON-compatible edit mapping.
    """
    result: dict[str, object] = {
        "patch_id": "patch-1",
        "request_id": "req-1",
        "sentence_id": "s-1",
        "operation": "replace",
        "original_text": "Karnataka",
        "replacement_text": "West Bengal",
        "evidence_ids": ["ev-1"],
        "error_type": "geographic",
        "explanation": "The source gives West Bengal.",
        "global_start": None,
        "global_end": None,
    }
    result.update(updates)
    return result


def test_proposer_returns_only_minimal_sentence_patch(tmp_path: Path) -> None:
    """Accept a one-entity replacement with no model-supplied offsets.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    proposer, _ = make_proposer(tmp_path, [edit()])
    outcome = proposer.propose(sentence(), request(), verification(), [evidence()])
    assert len(outcome.proposal.edits) == 1
    proposed = outcome.proposal.edits[0]
    assert proposed.original_text == "Karnataka"
    assert proposed.replacement_text == "West Bengal"
    assert proposed.global_start is None
    assert not outcome.errors


@pytest.mark.parametrize(
    "patch",
    [
        edit(original_text="August 15, 2024", replacement_text="September 5, 2025"),
        edit(original_text="Karnataka", replacement_text="West Bengal"),
        edit(original_text="15", replacement_text="5"),
        edit(
            operation="delete",
            original_text=" on August 15, 2024",
            replacement_text="",
        ),
        edit(
            operation="insert",
            original_text="",
            replacement_text="reportedly ",
            insertion_anchor="blocked",
        ),
    ],
)
def test_supported_edit_operations_are_structured(
    tmp_path: Path,
    patch: dict[str, object],
) -> None:
    """Accept bounded replacement, number, date, deletion, and insertion forms.

    Args:
        tmp_path: Temporary project root.
        patch: Structured operation fixture.

    Returns:
        None.
    """
    proposer, _ = make_proposer(tmp_path, [patch])
    outcome = proposer.propose(sentence(), request(), verification(), [evidence()])
    assert len(outcome.proposal.edits) == 1


def test_non_actionable_issue_never_calls_patch_model(tmp_path: Path) -> None:
    """Prefer no patch for unsupported or uncertain factual issues.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    proposer, transport = make_proposer(tmp_path, [edit()])
    unresolved = verification(
        verdict=Verdict.UNSUPPORTED,
        status=VerificationStatus.UNRESOLVED,
    )
    outcome = proposer.propose(sentence(), request(), unresolved, [evidence()])
    assert outcome.proposal.edits == ()
    assert transport.calls == []


@pytest.mark.parametrize(
    ("updates", "error_code"),
    [
        ({"sentence_id": "s-other"}, "patch_identity_mismatch"),
        ({"evidence_ids": ["invented"]}, "patch_unknown_evidence"),
        ({"error_type": "temporal"}, "patch_error_type_mismatch"),
        ({"original_text": "not in sentence"}, "patch_target_outside_sentence"),
        ({"global_start": 1, "global_end": 10}, "model_supplied_global_offsets"),
    ],
)
def test_unsafe_patch_is_rejected(
    tmp_path: Path,
    updates: dict[str, object],
    error_code: str,
) -> None:
    """Reject edits outside verified identity, evidence, type, or sentence bounds.

    Args:
        tmp_path: Temporary project root.
        updates: Unsafe edit fields.
        error_code: Required semantic rejection prefix.

    Returns:
        None.
    """
    proposer, _ = make_proposer(tmp_path, [edit(**updates)])
    outcome = proposer.propose(sentence(), request(), verification(), [evidence()])
    assert outcome.proposal.edits == ()
    assert outcome.errors == (f"{error_code}:patch-1",)


def test_whole_sentence_deletion_requires_explicit_unsupported_reason(
    tmp_path: Path,
) -> None:
    """Reject last-resort deletion without an unsupported-claim justification.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    patch = edit(
        operation=EditOperation.DELETE,
        original_text=SENTENCE_TEXT,
        replacement_text="",
        explanation="Shorten the article.",
    )
    proposer, _ = make_proposer(tmp_path, [patch])
    outcome = proposer.propose(sentence(), request(), verification(), [evidence()])
    assert outcome.errors == ("sentence_deletion_not_justified:patch-1",)
