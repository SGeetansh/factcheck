"""Offline tests for structured fact-check planning."""

from pathlib import Path

from article_rectifier.config import Settings
from article_rectifier.llm import LLMClient, MockLLMTransport
from article_rectifier.models import ArticleBatch, ErrorType, SentenceSpan
from article_rectifier.nodes.fact_planner import FactPlanner


def make_batch() -> ArticleBatch:
    """Create a two-sentence editable batch.

    Returns:
        Stable batch fixture.
    """
    first = SentenceSpan(
        sentence_id="s0001",
        text="The film opened on September 4, 2024.",
        start=0,
        end=37,
        paragraph_id="p0001",
    )
    second = SentenceSpan(
        sentence_id="s0002",
        text="It was blocked in Karnataka.",
        start=38,
        end=66,
        paragraph_id="p0001",
    )
    return ArticleBatch(batch_id="b0001", sentences=(first, second))


def make_client(tmp_path: Path, content: str) -> tuple[LLMClient, MockLLMTransport]:
    """Create a cached-offline client with one scripted response.

    Args:
        tmp_path: Temporary project root.
        content: Mock assistant JSON content.

    Returns:
        Client and inspectable mock transport.
    """
    response = {"choices": [{"message": {"content": content}}]}
    transport = MockLLMTransport([response])
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        max_llm_retries=0,
        _env_file=None,
    )
    return LLMClient(settings, transport), transport


def test_planner_returns_model_requests_and_deterministic_date(tmp_path: Path) -> None:
    """Keep a valid entity request and add a missed high-risk date request.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    content = """{"requests":[{"request_id":"req-place","sentence_id":"s0002",
    "claim_text":"It was blocked in Karnataka.",
    "verification_question":"Where was the film blocked?",
    "search_query":"film blocked theaters state",
    "error_category_hints":["geographic"],"sentence_start":38,"sentence_end":66}]}"""
    client, _ = make_client(tmp_path, content)
    outcome = FactPlanner(client).plan("article_004", make_batch())
    assert len(outcome.plan.requests) == 2
    assert outcome.plan.requests[0].error_category_hints == (ErrorType.GEOGRAPHIC,)
    assert outcome.plan.requests[1].error_category_hints == (ErrorType.TEMPORAL,)
    assert not outcome.errors


def test_planner_rejects_request_outside_editable_batch(tmp_path: Path) -> None:
    """Discard a schema-valid request with a fabricated sentence identifier.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    content = """{"requests":[{"request_id":"bad","sentence_id":"s9999",
    "claim_text":"Outside.","verification_question":"What?",
    "search_query":"outside","error_category_hints":[],
    "sentence_start":99,"sentence_end":107}]}"""
    client, _ = make_client(tmp_path, content)
    outcome = FactPlanner(client).plan("article_004", make_batch())
    assert len(outcome.plan.requests) == 1
    assert outcome.plan.requests[0].request_id.endswith("anchor-1")
    assert outcome.errors == ("invalid_request_boundary:bad",)


def test_planner_failure_still_checks_obvious_numbers(tmp_path: Path) -> None:
    """Use deterministic anchors when structured planning fails.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    responses = [
        {"choices": [{"message": {"content": "invalid"}}]},
        {"choices": [{"message": {"content": "still invalid"}}]},
    ]
    transport = MockLLMTransport(responses)
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        max_llm_retries=0,
        _env_file=None,
    )
    outcome = FactPlanner(LLMClient(settings, transport)).plan(
        "article_004", make_batch()
    )
    assert len(outcome.plan.requests) == 1
    assert outcome.errors


def test_prompt_excludes_annotations_and_source_truth(tmp_path: Path) -> None:
    """Send only article identity, batch text, context, and schema to planning.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    client, transport = make_client(tmp_path, '{"requests":[]}')
    FactPlanner(client).plan("article_004", make_batch())
    messages = transport.calls[0]["messages"]
    serialized = str(messages)
    assert "Error Annotations" not in serialized
    assert "West Bengal" not in serialized
    assert "Karnataka" in serialized


def test_planner_normalizes_a_paraphrase_with_valid_sentence_offsets(
    tmp_path: Path,
) -> None:
    """Retain a paraphrased claim only within its verified editable sentence.

    Args:
        tmp_path: Temporary project root.

    Returns:
        None.
    """
    content = """{"requests":[{"request_id":"req-place","sentence_id":"s0002",
    "claim_text":"The film faced a ban in Karnataka",
    "verification_question":"Where was it blocked?",
    "search_query":"film blocked theaters state",
    "error_category_hints":["geographic"],
    "sentence_start":38,"sentence_end":66}]}"""
    client, _ = make_client(tmp_path, content)
    outcome = FactPlanner(client).plan("article_004", make_batch())
    assert not outcome.errors
    assert outcome.plan.requests[0].request_id == "req-place"
    assert outcome.plan.requests[0].claim_text == "It was blocked in Karnataka."
