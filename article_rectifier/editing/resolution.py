"""Account for verified corrections using accepted edits, not request identity."""

from article_rectifier.models.article import SentenceSpan
from article_rectifier.models.edits import SurgicalEdit
from article_rectifier.models.results import (
    FactCorrection,
    PatchStatus,
    Verdict,
    VerificationResult,
)
from article_rectifier.models.state import BatchState


def correction_applied(
    fact: FactCorrection, sentence: SentenceSpan, edits: list[SurgicalEdit]
) -> bool:
    """Check the exact final value at a unique original fact span.

    Args:
        fact: Structured, verified value substitution.
        sentence: Immutable sentence containing the original fact.
        edits: Final accepted edits with original article offsets.

    Returns:
        Whether accepted edits implement this entire substitution. Ambiguous or
        crossing spans remain unresolved; this is not a new factual judgment.
    """
    if not fact.original_value or sentence.text.count(fact.original_value) != 1:
        return False
    start = sentence.start + sentence.text.index(fact.original_value)
    end = start + len(fact.original_value)
    relevant = []
    for edit in edits:
        left, right = edit.global_start, edit.global_end
        if edit.sentence_id != sentence.sentence_id or left is None or right is None:
            continue
        if left < end and right > start:
            if left < start or right > end:
                return False
            relevant.append(edit)
    value = fact.original_value
    for edit in sorted(relevant, key=lambda item: item.global_start, reverse=True):
        left, right = edit.global_start - start, edit.global_end - start
        value = value[:left] + edit.replacement_text + value[right:]
    return bool(relevant) and value == fact.correct_value


def unresolved_results(state: BatchState) -> list[VerificationResult]:
    """Find unimplemented corrections and judgments lacking source support.

    Args:
        state: Completed batch with final accepted patches and original requests.

    Returns:
        Unresolved judgments. All facts in a structured correction must be applied,
        even if one patch already has the same request ID.
    """
    edits = [
        item.edit
        for item in state["patch_results"]
        if item.status is PatchStatus.ACCEPTED
    ]
    requests = {item.request_id: item for item in state["fact_requests"]}
    sentences = {item.sentence_id: item for item in state["batch"].sentences}
    unresolved = []
    for result in state["verification_results"]:
        if result.verdict in {Verdict.UNSUPPORTED, Verdict.UNCERTAIN}:
            unresolved.append(result)
        elif result.verdict is Verdict.CONTRADICTED:
            request = requests.get(result.request_id)
            sentence = sentences.get(request.sentence_id) if request else None
            complete = (
                bool(sentence)
                and all(
                    correction_applied(fact, sentence, edits)
                    for fact in result.corrections
                )
                if result.corrections
                else any(edit.request_id == result.request_id for edit in edits)
            )
            if not complete:
                unresolved.append(result)
    return unresolved
