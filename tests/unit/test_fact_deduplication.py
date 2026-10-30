"""Keep distinct facts while merging duplicate questions about the same claim."""

from article_rectifier.models.claims import FactCheckRequest
from article_rectifier.nodes.fact_planner import _merge_requests


def request(identifier: str, claim: str, question: str) -> FactCheckRequest:
    """Build a request for a fixed sentence without model calls."""
    return FactCheckRequest(
        request_id=identifier,
        sentence_id="s1",
        claim_text=claim,
        verification_question=question,
        search_query=claim,
        sentence_start=0,
        sentence_end=100,
    )


def test_same_claim_questions_merge_without_losing_distinct_checks() -> None:
    """Keep both questions under one identity while retaining a different claim."""
    requests = [
        request("r1", "Price is 20 in 2025", "What price?"),
        request("r2", "Price is 20 in 2025", "Which year?"),
        request("r3", "Price is 20", "What price?"),
    ]
    result = _merge_requests(requests, [])
    assert len(result.requests) == 2
    assert result.requests[0].request_id == "r1"
    assert "What price?" in result.requests[0].verification_question
    assert "Which year?" in result.requests[0].verification_question
