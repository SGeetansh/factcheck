"""Resolve model-proposed exact text to immutable article offsets."""

import re

from article_rectifier.models.article import SentenceSpan
from article_rectifier.models.edits import EditOperation, SurgicalEdit


def find_unique_span(text: str, target: str, start: int, end: int) -> tuple[int, int]:
    """Locate one exact occurrence within a permitted region.

    Args:
        text: Complete original article text.
        target: Exact text to locate.
        start: Inclusive lower search boundary.
        end: Exclusive upper search boundary.

    Returns:
        A tuple containing the start and end offsets.

    Raises:
        ValueError: If the span is absent or ambiguous.
    """
    if not target:
        raise ValueError("Exact span target cannot be empty")
    first = text.find(target, start, end)
    if first < 0:
        raise ValueError("Patch target is absent from permitted sentence")
    second = text.find(target, first + 1, end)
    if second >= 0:
        raise ValueError("Patch target is ambiguous within permitted sentence")
    return first, first + len(target)


def resolve_edit(
    article_text: str,
    sentence: SentenceSpan,
    edit: SurgicalEdit,
) -> SurgicalEdit:
    """Resolve one proposed edit to exact global original-text offsets.

    Args:
        article_text: Immutable sanitized generated article.
        sentence: Only sentence permitted for the edit.
        edit: Proposed edit without trusted global offsets.

    Returns:
        Copy of the edit containing Python-resolved global offsets.

    Raises:
        ValueError: If identity, source text, target, or insertion is unsafe.
    """
    if edit.sentence_id != sentence.sentence_id:
        raise ValueError("Patch sentence identity does not match permitted sentence")
    if article_text[sentence.start : sentence.end] != sentence.text:
        raise ValueError("Sentence span does not match immutable article text")
    if edit.global_start is not None or edit.global_end is not None:
        raise ValueError("Proposed edits must not contain trusted global offsets")
    if edit.operation is EditOperation.INSERT:
        anchor_start, anchor_end = find_unique_span(
            article_text, edit.insertion_anchor or "", sentence.start, sentence.end
        )
        position = anchor_end if edit.insert_after_anchor else anchor_start
        return edit.model_copy(
            update={"global_start": position, "global_end": position}
        )
    start, end = find_unique_span(
        article_text, edit.original_text, sentence.start, sentence.end
    )
    if re.fullmatch(r"\d+,", edit.original_text) and re.fullmatch(
        r"\d+", edit.replacement_text
    ):
        edit = edit.model_copy(update={"replacement_text": edit.replacement_text + ","})
    return edit.model_copy(update={"global_start": start, "global_end": end})
