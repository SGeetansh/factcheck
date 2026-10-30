"""Remove trailing development metadata without changing article content."""

import json
import re

ANNOTATION_MARKER = re.compile(
    r"(?im)^(?:\*\*|#{1,6}\s*)?Error Annotations:(?:\*\*)?[ \t]*$"
)


def _is_annotation_payload(payload: str) -> bool:
    """Check whether trailing text has the supplied annotation structure.

    Args:
        payload: Text after a possible annotation heading.

    Returns:
        True for a JSON list of annotation-like objects.
    """
    candidate = payload.removesuffix("```").rstrip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return (
            candidate.startswith("[")
            and candidate.endswith("]")
            and '"location"' in candidate
            and '"correction"' in candidate
        )
    if not isinstance(parsed, list):
        return False
    return all(
        isinstance(item, dict) and {"error", "correction"}.issubset(item)
        for item in parsed
    )


def sanitize_generated_article(text: str) -> str:
    """Remove only a valid trailing error-annotation metadata block.

    Args:
        text: Generated article, possibly followed by development annotations.

    Returns:
        Original text before the metadata marker, byte-for-byte as decoded.
    """
    matches = list(ANNOTATION_MARKER.finditer(text))
    for match in reversed(matches):
        payload = text[match.end() :].strip()
        if _is_annotation_payload(payload):
            return text[: match.start()]
    return text


def sanitize_source_article(text: str) -> str:
    """Remove a source-administration marker while preserving source metadata.

    Args:
        text: Complete supplied source article.

    Returns:
        Source text without a standalone `Source Article(s):` label.
    """
    marker = re.compile(r"(?im)^Source Article\(s\):[ \t]*(?:\r?\n)?")
    return marker.sub("", text, count=1)
