"""Tests for safe sanitization, sentence offsets, and batching."""

from article_rectifier.preprocessing.batching import create_batches
from article_rectifier.preprocessing.sanitizer import (
    sanitize_generated_article,
    sanitize_source_article,
)
from article_rectifier.preprocessing.sentence_splitter import (
    reconstruct_from_spans,
    split_sentences,
)


def test_sanitizer_removes_only_valid_trailing_annotations() -> None:
    """Remove supplied metadata while preserving every preceding character.

    Returns:
        None.
    """
    article = "## Heading\n\nArticle text.\n"
    metadata = (
        "**Error Annotations:**\n"
        '[{"error": "old", "correction": "new", "location": "p1"}]'
    )
    assert sanitize_generated_article(article + metadata) == article


def test_sanitizer_preserves_legitimate_marker_text() -> None:
    """Keep prose and malformed metadata that do not match the supplied block.

    Returns:
        None.
    """
    prose = "The guide mentions **Error Annotations:** in one sentence."
    heading = "Article.\n\n**Error Annotations:**\nThis is ordinary prose."
    assert sanitize_generated_article(prose) == prose
    assert sanitize_generated_article(heading) == heading


def test_sanitizer_removes_supplied_malformed_json_and_code_fence() -> None:
    """Remove recognizable trailing metadata even when its generator escaped it badly.

    Returns:
        None.
    """
    malformed = (
        "Article.\n**Error Annotations:**\n"
        '[{"location":"p1","error": \\"old\\", "correction":"new"}]'
    )
    fenced_empty = "Article.\n**Error Annotations:**\n[]\n```"
    assert sanitize_generated_article(malformed) == "Article.\n"
    assert sanitize_generated_article(fenced_empty) == "Article.\n"


def test_source_sanitizer_keeps_published_metadata() -> None:
    """Remove only the administrative marker from source content.

    Returns:
        None.
    """
    source = "Source Article(s):\nPublished: Sep 5, 2025\n\nReport text."
    assert sanitize_source_article(source) == "Published: Sep 5, 2025\n\nReport text."


def test_sentence_splitter_handles_common_boundary_traps() -> None:
    """Keep abbreviations and decimals while splitting quoted sentences.

    Returns:
        None.
    """
    text = (
        "## Report\n\n"
        'Dr. Rao said, "Revenue rose 3.5%." The U.S. unit agreed. '
        "Was it final? Yes—it was\uff01"
    )
    spans = split_sentences(text)
    assert [span.text for span in spans] == [
        "## Report",
        'Dr. Rao said, "Revenue rose 3.5%."',
        "The U.S. unit agreed.",
        "Was it final?",
        "Yes—it was\uff01",
    ]
    assert reconstruct_from_spans(text, spans) == text


def test_sentence_offsets_support_unicode_and_duplicate_sentences() -> None:
    """Keep Python string offsets correct for Unicode and repeated text.

    Returns:
        None.
    """
    text = "Price was ₹50. Price was ₹50. “Done.”"
    spans = split_sentences(text)
    assert len(spans) == 3
    assert spans[0].text == spans[1].text
    for span in spans:
        assert text[span.start : span.end] == span.text
    assert spans[0].start != spans[1].start


def test_empty_input_has_no_sentences_or_batches() -> None:
    """Handle empty articles without inventing content.

    Returns:
        None.
    """
    assert split_sentences("") == []
    assert create_batches([]) == []


def test_three_sentence_batches_have_read_only_neighbors() -> None:
    """Cover each sentence once and attach only adjacent context.

    Returns:
        None.
    """
    text = "One. Two. Three. Four. Five. Six. Seven."
    sentences = split_sentences(text)
    batches = create_batches(sentences)
    assert [len(batch.sentences) for batch in batches] == [3, 3, 1]
    assert batches[0].context_before is None
    assert batches[0].context_after == sentences[3]
    assert batches[1].context_before == sentences[2]
    assert batches[1].context_after == sentences[6]
    assert batches[2].context_before == sentences[5]
    assert batches[2].context_after is None
    covered = [item for batch in batches for item in batch.sentences]
    assert covered == sentences


def test_headings_paragraphs_and_spacing_reconstruct_exactly() -> None:
    """Preserve headings, multiline paragraphs, and formatting gaps.

    Returns:
        None.
    """
    text = "## Heading\n\nFirst sentence.  Second sentence.\nStill second.\n\nLast"
    spans = split_sentences(text)
    assert reconstruct_from_spans(text, spans) == text
    assert spans[0].paragraph_id != spans[1].paragraph_id
