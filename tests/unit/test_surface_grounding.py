"""Known equivalent surface forms must not block source-supported corrections."""

import pytest

from article_rectifier.models import EvidenceSpan, RetrievalStrategy, SurgicalEdit
from article_rectifier.nodes.validator import _replacement_is_grounded


@pytest.mark.parametrize(
    "replacement, source, expected",
    [
        ("vapor chamber cooling system", "A vapour chamber cooling system.", True),
        ("3,000 nits", "Peak brightness is 3,000nits.", True),
        ("30,000 nits", "Peak brightness is 3,000nits.", False),
        ("₹", "The price is Rs. 1,34,900.", True),
        ("$", "The price is Rs. 1,34,900.", False),
        ("graphene cooling system", "A vapour chamber cooling system.", False),
    ],
)
def test_narrow_surface_equivalence(
    replacement: str, source: str, expected: bool
) -> None:
    """Accept known notation differences, while rejecting wrong values and materials."""
    evidence = EvidenceSpan(
        evidence_id="e1",
        source_article_id="a1",
        text=source,
        start=0,
        end=len(source),
        retrieval_score=1,
        retrieval_strategy=RetrievalStrategy.BM25,
    )
    edit = SurgicalEdit(
        patch_id="p1",
        request_id="r1",
        sentence_id="s1",
        original_text="incorrect",
        replacement_text=replacement,
        evidence_ids=("e1",),
        error_type="other",
        explanation="Source comparison.",
    )
    assert _replacement_is_grounded(edit, [evidence]) is expected
