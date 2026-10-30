"""Merge targeted retry results while retaining trusted accepted offsets."""

from article_rectifier.editing.diff_guard import (
    apply_resolved_edits,
    conflicting_patch_ids,
)
from article_rectifier.models.results import PatchResult, PatchStatus


def merge_retry_results(
    original: str,
    previous: list[PatchResult],
    current: list[PatchResult],
    retry_count: int,
    max_retries: int,
) -> tuple[str, list[PatchResult]]:
    """Keep accepted patches fixed and retry only new failed dispositions.

    Args:
        original: Immutable article text.
        previous: Prior dispositions from this batch.
        current: Newly executed proposed patches.
        retry_count: Number of previous retries.
        max_retries: Configured retry bound.

    Returns:
        Exact reconstruction and final combined dispositions.
    """
    retained = (
        [item for item in previous if item.status is PatchStatus.ACCEPTED]
        if retry_count
        else []
    )
    conflicts = conflicting_patch_ids(
        [
            item.edit
            for item in [*retained, *current]
            if item.status is PatchStatus.ACCEPTED
        ]
    )
    merged = list(retained)
    for item in current:
        if item.status is PatchStatus.ACCEPTED and item.edit.patch_id in conflicts:
            item = item.model_copy(
                update={
                    "status": PatchStatus.REJECTED,
                    "reason": "retry overlaps a retained accepted patch",
                }
            )
        if item.status is PatchStatus.REJECTED:
            item = item.model_copy(
                update={
                    "status": (
                        PatchStatus.RETRYABLE
                        if retry_count < max_retries
                        else PatchStatus.UNRESOLVED
                    )
                }
            )
        merged.append(item)
    edits = [item.edit for item in merged if item.status is PatchStatus.ACCEPTED]
    return apply_resolved_edits(original, edits), merged
