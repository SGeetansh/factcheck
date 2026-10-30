"""Tests that protect the human-corrected reference fixtures."""

import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FIXTURE_ROOT = PROJECT_ROOT / "evaluate" / "fixtures" / "human_references"


def sha256(path: Path) -> str:
    """Calculate a file's SHA-256 digest.

    Args:
        path: File to hash.

    Returns:
        Lowercase hexadecimal digest.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_human_references_match_original_files() -> None:
    """Require all ten fixture copies to match their supplied originals.

    Returns:
        None.
    """
    manifest_lines = (FIXTURE_ROOT / "SHA256SUMS").read_text().splitlines()
    expected = dict(line.split(maxsplit=1) for line in manifest_lines)
    assert len(expected) == 10
    for digest, filename in expected.items():
        original = PROJECT_ROOT / "rectified_articles" / filename
        fixture = FIXTURE_ROOT / filename
        assert sha256(original) == digest
        assert sha256(fixture) == digest
