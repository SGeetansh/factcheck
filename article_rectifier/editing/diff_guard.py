"""Validate patch conflicts and exact reconstruction invariants."""

from article_rectifier.models.edits import SurgicalEdit


def _offsets(edit: SurgicalEdit) -> tuple[int, int]:
    """Return resolved offsets or fail closed.

    Args:
        edit: Python-resolved edit.

    Returns:
        Inclusive start and exclusive end.

    Raises:
        ValueError: If offsets were not resolved.
    """
    if edit.global_start is None or edit.global_end is None:
        raise ValueError("Patch offsets are unresolved")
    return edit.global_start, edit.global_end


def conflicting_patch_ids(edits: list[SurgicalEdit]) -> set[str]:
    """Find overlapping replacements and ambiguous co-located insertions.

    Args:
        edits: Python-resolved proposed edits.

    Returns:
        IDs of all patches participating in at least one conflict.
    """
    conflicts: set[str] = set()
    ordered = sorted(edits, key=lambda item: (*_offsets(item), item.patch_id))
    for index, left in enumerate(ordered):
        left_start, left_end = _offsets(left)
        for right in ordered[index + 1 :]:
            right_start, right_end = _offsets(right)
            if right_start > left_end:
                break
            co_insert = left_start == left_end == right_start == right_end
            overlaps = right_start < left_end and left_start < right_end
            touches_insert = (
                left_start == left_end and right_start <= left_start < right_end
            ) or (right_start == right_end and left_start <= right_start < left_end)
            if co_insert or overlaps or touches_insert:
                conflicts.update((left.patch_id, right.patch_id))
    return conflicts


def replace_span(
    text: str,
    start: int,
    end: int,
    expected: str,
    replacement: str,
) -> str:
    """Replace one verified character span.

    Args:
        text: Original text.
        start: Inclusive start offset.
        end: Exclusive end offset.
        expected: Expected text at the specified range.
        replacement: New text for the specified range.

    Returns:
        Text with only the specified replacement applied.

    Raises:
        ValueError: If offsets or original text do not match.
    """
    if not 0 <= start <= end <= len(text):
        raise ValueError("Invalid patch offsets")
    if text[start:end] != expected:
        raise ValueError("Patch does not match original text")
    return text[:start] + replacement + text[end:]


def apply_resolved_edits(text: str, edits: list[SurgicalEdit]) -> str:
    """Apply non-conflicting edits from right to left against original offsets.

    Args:
        text: Immutable original article.
        edits: Authorized Python-resolved edits.

    Returns:
        Article reconstructed from the original and exact patches.

    Raises:
        ValueError: If edits conflict or do not match the original.
    """
    conflicts = conflicting_patch_ids(edits)
    if conflicts:
        raise ValueError(f"Conflicting patches: {','.join(sorted(conflicts))}")
    output = text
    for edit in sorted(edits, key=lambda item: _offsets(item)[0], reverse=True):
        start, end = _offsets(edit)
        output = replace_span(
            output, start, end, edit.original_text, edit.replacement_text
        )
    return output
