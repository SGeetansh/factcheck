"""Offline tests for source-grounded factual verification."""

import json
from pathlib import Path

import pytest

from article_rectifier.config import Settings
from article_rectifier.llm import LLMClient, MockLLMTransport
from article_rectifier.models import (
    ErrorType,
    EvidenceSpan,
    FactCheckRequest,
    RetrievalStrategy,
    Verdict,
    VerificationStatus,
)
from article_rectifier.nodes.evidence_verifier import EvidenceVerifier

SOURCE = "Most theater owners in West Bengal refused to screen the film."


def make_request() -> FactCheckRequest:
    """Create a geographic fact-check request.

    Returns:
        Valid request fixture.
    """
    return FactCheckRequest(
        request_id="req-1",
        sentence_id="s-1",
        claim_text="The film was blocked in Karnataka.",
        verification_question="Where was the film blocked?",
        search_query="film theater owners blocked state",
        error_category_hints=(ErrorType.GEOGRAPHIC,),
        sentence_start=0,
        sentence_end=34,
    )


def make_evidence(text: str = SOURCE) -> EvidenceSpan:
    """Create exact evidence over a supplied source string.

    Args:
        text: Evidence and complete source text.

    Returns:
        Valid evidence span fixture.
    """
    return EvidenceSpan(
        evidence_id="ev-1",
        source_article_id="article-4",
        text=text,
        start=0,
        end=len(text),
        retrieval_score=3.0,
        retrieval_strategy=RetrievalStrategy.BM25,
    )


def make_verifier(
    tmp_path: Path,
    result: dict[str, object],
) -> tuple[EvidenceVerifier, MockLLMTransport]:
    """Create a verifier with one scripted JSON response.

    Args:
        tmp_path: Temporary project root.
        result: Structured verification response.

    Returns:
        Verifier and inspectable mock transport.
    """
    response = {"choices": [{"message": {"content": json.dumps(result)}}]}
    transport = MockLLMTransport([response])
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        max_llm_retries=0,
        _env_file=None,
    )
    return EvidenceVerifier(LLMClient(settings, transport)), transport


def contradicted_result(**updates: object) -> dict[str, object]:
    """Create a grounded contradiction response with overrides.

    Args:
        **updates: Response fields to replace.

    Returns:
        JSON-compatible result mapping.
    """
    result: dict[str, object] = {
        "request_id": "req-1",
        "verdict": "contradicted",
        "evidence_ids": ["ev-1"],
        "explanation": "The source identifies West Bengal.",
        "error_types": ["geographic"],
        "proposed_factual_correction": "West Bengal",
        "status": "correction_available",
    }
    result.update(updates)
    return result


def test_contradiction_with_grounded_correction_is_actionable(tmp_path: Path) -> None:
    """Accept a corrected fact whose tokens occur in cited evidence.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    verifier, _ = make_verifier(tmp_path, contradicted_result())
    outcome = verifier.verify(make_request(), [make_evidence()], SOURCE)
    assert outcome.result.verdict is Verdict.CONTRADICTED
    assert outcome.result.status is VerificationStatus.CORRECTION_AVAILABLE
    assert outcome.result.proposed_factual_correction == "West Bengal"
    assert not outcome.errors


@pytest.mark.parametrize(
    ("verdict", "status"),
    [
        ("supported", "no_change"),
        ("unsupported", "unresolved"),
        ("uncertain", "unresolved"),
    ],
)
def test_non_actionable_verdicts_remain_distinct(
    tmp_path: Path,
    verdict: str,
    status: str,
) -> None:
    """Preserve supported, unsupported, and uncertain classifications.

    Args:
        tmp_path: Temporary project root.
        verdict: Mock verdict value.
        status: Required compatible status.

    Returns:
        None.
    """
    result = {
        "request_id": "req-1",
        "verdict": verdict,
        "evidence_ids": ["ev-1"] if verdict == "supported" else [],
        "explanation": "A bounded judgment.",
        "error_types": [],
        "proposed_factual_correction": None,
        "status": status,
    }
    verifier, _ = make_verifier(tmp_path, result)
    outcome = verifier.verify(make_request(), [make_evidence()], SOURCE)
    assert outcome.result.verdict.value == verdict
    assert outcome.result.status.value == status


def test_missing_or_irrelevant_evidence_skips_llm(tmp_path: Path) -> None:
    """Return unsupported before model use when passages do not answer the topic.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    verifier, transport = make_verifier(tmp_path, contradicted_result())
    empty = verifier.verify(make_request(), [], SOURCE)
    unrelated_text = "A recipe uses flour and water."
    unrelated = verifier.verify(
        make_request(), [make_evidence(unrelated_text)], unrelated_text
    )
    assert empty.result.verdict is Verdict.UNSUPPORTED
    assert unrelated.result.verdict is Verdict.UNSUPPORTED
    assert transport.calls == []


@pytest.mark.parametrize(
    ("updates", "expected_error"),
    [
        ({"request_id": "other"}, "request_id_mismatch"),
        ({"evidence_ids": ["invented"]}, "unknown_evidence_id"),
        (
            {"proposed_factual_correction": "Kerala"},
            "correction_not_in_evidence",
        ),
    ],
)
def test_semantically_unsafe_verdict_is_downgraded(
    tmp_path: Path,
    updates: dict[str, object],
    expected_error: str,
) -> None:
    """Downgrade fabricated identities, citations, and corrections.

    Args:
        tmp_path: Temporary project root.
        updates: Unsafe response fields.
        expected_error: Required semantic error code.

    Returns:
        None.
    """
    verifier, _ = make_verifier(tmp_path, contradicted_result(**updates))
    outcome = verifier.verify(make_request(), [make_evidence()], SOURCE)
    assert outcome.result.verdict is Verdict.UNCERTAIN
    assert outcome.result.status is VerificationStatus.UNRESOLVED
    assert outcome.errors == (expected_error,)


@pytest.mark.parametrize(
    "error_type",
    ["negation", "magnitude", "causal", "attribution", "relationship"],
)
def test_error_classifications_are_preserved(
    tmp_path: Path,
    error_type: str,
) -> None:
    """Keep supported error categories on an actionable contradiction.

    Args:
        tmp_path: Temporary project root.
        error_type: Fixed schema category.

    Returns:
        None.
    """
    verifier, _ = make_verifier(tmp_path, contradicted_result(error_types=[error_type]))
    outcome = verifier.verify(make_request(), [make_evidence()], SOURCE)
    assert outcome.result.error_types[0].value == error_type
