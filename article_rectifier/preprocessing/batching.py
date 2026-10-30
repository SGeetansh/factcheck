"""Create fixed-size editable batches with read-only sentence context."""

from article_rectifier.models.article import ArticleBatch, SentenceSpan


def create_batches(
    sentences: list[SentenceSpan],
    batch_size: int = 3,
) -> list[ArticleBatch]:
    """Group sentence spans without changing their immutable offsets.

    Args:
        sentences: Ordered sentence spans from one sanitized article.
        batch_size: Maximum editable sentences per batch.

    Returns:
        Ordered batches with at most one neighboring sentence on each side.

    Raises:
        ValueError: If batch size is not positive.
    """
    if batch_size < 1:
        raise ValueError("Batch size must be positive")
    batches: list[ArticleBatch] = []
    for start in range(0, len(sentences), batch_size):
        end = min(start + batch_size, len(sentences))
        batches.append(
            ArticleBatch(
                batch_id=f"b{len(batches) + 1:04d}",
                sentences=tuple(sentences[start:end]),
                context_before=sentences[start - 1] if start else None,
                context_after=sentences[end] if end < len(sentences) else None,
            )
        )
    return batches
