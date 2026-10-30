"""Regression tests for shared verification and exact fact-pair editing."""

import json
from pathlib import Path

from article_rectifier.config import Settings
from article_rectifier.llm import LLMClient, MockLLMTransport
from article_rectifier.models.article import ArticleInput
from article_rectifier.pipeline import ArticleRectificationPipeline


def response(value: dict[str, object]) -> dict[str, object]:
    """Create a structured transport fixture with observable usage."""
    return {
        "choices": [{"message": {"content": json.dumps(value)}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20},
    }


def test_batch_verification_builds_small_patches_without_proposer_calls(
    tmp_path: Path,
) -> None:
    """Verify two sentences once, then construct two exact replacements in Python."""
    source = "Price fell by ₹5,000. Engines are 1,200cc."
    original = "Price fell by ₹2,000. Engines are 1,000cc."
    evidence_id = "src-fixture-0001"
    results = [
        {
            "request_id": f"b0001-s000{index}-anchor-1",
            "verdict": "contradicted",
            "evidence_ids": [evidence_id],
            "explanation": "The source gives a different number.",
            "error_types": ["numerical"],
            "status": "correction_available",
            "corrections": [
                {
                    "original_value": old,
                    "correct_value": new,
                    "evidence_ids": [evidence_id],
                    "error_type": "numerical",
                }
            ],
        }
        for index, old, new in [(1, "2,000", "5,000"), (2, "1,000", "1,200")]
    ]
    review = response(
        {
            "verdict": "supported",
            "evidence_ids": [evidence_id],
            "explanation": "The corrected numeric claim matches the source.",
            "failed_patch_ids": [],
        }
    )
    transport = MockLLMTransport(
        [response({"requests": []}), response({"results": results}), review, review]
    )
    settings = Settings(
        LLM_API_KEY="test-key",
        project_root=tmp_path,
        enable_llm_cache=False,
        _env_file=None,
    )
    pipeline = ArticleRectificationPipeline(settings, LLMClient(settings, transport))
    result = pipeline.rectify(
        ArticleInput(article_id="fixture", source_text=source, generated_text=original)
    )
    assert result.output_text == source
    assert len(result.accepted_edits) == 2
    assert len(transport.calls) == 4
    assert [record.schema_name for record in result.llm_records] == [
        "FactCheckPlan",
        "VerificationBatch",
        "SentenceReview",
        "SentenceReview",
    ]
    assert {edit.original_text for edit in result.accepted_edits} == {"2,000", "1,000"}
