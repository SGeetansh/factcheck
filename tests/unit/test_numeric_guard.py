"""Check exact quantity equivalence without treating source words as entailment."""

from decimal import Decimal

from article_rectifier.models import EvidenceSpan, RetrievalStrategy, SurgicalEdit
from article_rectifier.nodes.numeric_guard import numeric_error, numeric_values


def test_indian_and_western_numeric_scales_are_exact() -> None:
    """Equivalent printed magnitudes must produce the same Decimal value."""
    assert numeric_values("₹5,77,000") == numeric_values("Rs 5.77 lakh")
    assert numeric_values("₹20.8L") == numeric_values("₹20,80,000")
    assert numeric_values("1.2 crore") == numeric_values("12 million")
    assert numeric_values("18% in 2025") == {Decimal(18), Decimal(2025)}


def test_lm_price_claim_can_reach_semantic_review_but_wrong_number_cannot() -> None:
    """Allow grammatical paraphrase while blocking an unsupported replacement amount."""
    source = "The LM 350h has a price reduction of up to Rs 5.77 lakh."
    evidence = EvidenceSpan(
        evidence_id="e",
        source_article_id="a",
        text=source,
        start=0,
        end=len(source),
        retrieval_score=1,
        retrieval_strategy=RetrievalStrategy.BM25,
    )
    edit = SurgicalEdit(
        patch_id="p",
        request_id="r",
        sentence_id="s",
        original_text="will see no change",
        replacement_text="will be cheaper by up to ₹5,77,000",
        evidence_ids=("e",),
        error_type="numerical",
        explanation="Source-grounded amount.",
    )
    assert numeric_error(edit, [evidence]) is None
    wrong = edit.model_copy(update={"replacement_text": "will be cheaper by ₹7,77,000"})
    assert numeric_error(wrong, [evidence]) is not None
