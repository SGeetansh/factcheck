"""Independent patch precision and count metrics."""

from collections import Counter

from evaluate.models import DetectedIssue, PatchCounts, PatchObservation, ScoreResult
from evaluate.preservation import unavailable_score


def edit_precision(
    patches: list[PatchObservation],
    *,
    applied_only: bool,
) -> ScoreResult:
    """Calculate precision only from independently reviewed patches.

    Args:
        patches: Patch observations with optional justification judgments.
        applied_only: Restrict the metric to applied patches when true.

    Returns:
        Precision over reviewed patches, or unavailable without review.
    """
    name = "applied_edit_precision" if applied_only else "proposed_edit_precision"
    candidates = [patch for patch in patches if patch.applied or not applied_only]
    reviewed = [patch for patch in candidates if patch.justified is not None]
    if not reviewed:
        return unavailable_score(
            name, "No patches have independent justification labels."
        )
    justified = sum(patch.justified is True for patch in reviewed)
    return ScoreResult(
        name=name,
        value=justified / len(reviewed),
        numerator=justified,
        denominator=len(reviewed),
        note=(
            "Unreviewed patches are excluded; agent self-approval is not "
            "justification."
        ),
    )


def count_patches(
    patches: list[PatchObservation],
    *,
    unresolved_issues: int,
    retrieval_failures: int,
) -> PatchCounts:
    """Count proposed, applied, rejected, and unresolved work.

    Args:
        patches: Proposed patch observations.
        unresolved_issues: Independently observed unresolved issue count.
        retrieval_failures: Observed evidence retrieval failure count.

    Returns:
        Deterministic operational counts.
    """
    applied = sum(patch.applied for patch in patches)
    return PatchCounts(
        proposed=len(patches),
        applied=applied,
        rejected=len(patches) - applied,
        unresolved_issues=unresolved_issues,
        retrieval_failures=retrieval_failures,
    )


def issue_type_counts(issues: list[DetectedIssue]) -> dict[str, int]:
    """Count detected issues by their reported category.

    Args:
        issues: Detected issue observations.

    Returns:
        Sorted mapping from error type to observed count.
    """
    counts = Counter(issue.error_type or "unclassified" for issue in issues)
    return dict(sorted(counts.items()))
