"""Reduce authorized contextual replacements to independent token changes."""

import re
from difflib import SequenceMatcher

from article_rectifier.models.edits import EditOperation, SurgicalEdit

TOKEN = re.compile(r"\w+(?:[.,]\w+)*|\s+|[^\w\s]")


def changed_fragments(original: str, replacement: str) -> list[tuple[int, int, str]]:
    """Locate token-aligned replacements while preserving exact whitespace.

    Args:
        original: Authorized original span.
        replacement: Proposed replacement span.

    Returns:
        Independent changes, or the full change when insertions need context.
    """
    old = list(TOKEN.finditer(original))
    new = list(TOKEN.finditer(replacement))
    changes = []
    matcher = SequenceMatcher(
        None, [x.group() for x in old], [x.group() for x in new], autojunk=False
    )
    for operation, a, b, c, d in matcher.get_opcodes():
        if operation == "equal":
            continue
        if a == b:
            return [(0, len(original), replacement)]
        text = replacement[new[c].start() : new[d - 1].end()] if c != d else ""
        changes.append((old[a].start(), old[b - 1].end(), text))
    return changes


def minimize_resolved_edit(edit: SurgicalEdit) -> list[SurgicalEdit]:
    """Split a Python-resolved replacement without changing its final output.

    Args:
        edit: Authorized edit with trusted original offsets.

    Returns:
        Smaller resolved edits. Context used for disambiguation stays unchanged.
    """
    if edit.operation is not EditOperation.REPLACE or edit.global_start is None:
        return [edit]
    changes = changed_fragments(edit.original_text, edit.replacement_text)
    return [
        edit.model_copy(
            update={
                "patch_id": (
                    edit.patch_id
                    if len(changes) == 1
                    else f"{edit.patch_id}:part{index}"
                ),
                "original_text": edit.original_text[start:end],
                "replacement_text": replacement,
                "operation": (
                    EditOperation.REPLACE if replacement else EditOperation.DELETE
                ),
                "global_start": edit.global_start + start,
                "global_end": edit.global_start + end,
            }
        )
        for index, (start, end, replacement) in enumerate(changes, 1)
    ]
