"""Check retrieval scores across Python processes with different hash seeds."""

import os
import subprocess
import sys


def test_scores_are_identical_across_hash_seeds() -> None:
    """Identical ranking inputs must yield byte-identical cache-affecting scores."""
    code = """
import json
from article_rectifier.retrieval.source_index import build_source_index
from article_rectifier.retrieval.retriever import search_source
index = build_source_index("The family paid an agent for the journey.\\n\\n"
                           "The journey cost his family around 45 lakh.\\n\\n"
                           "The man lives in a different district.", "test")
print(json.dumps([x.model_dump(mode="json") for x in search_source(
    "family agent journey cost man district", index)], sort_keys=True))
"""
    outputs = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": str(seed)},
            text=True,
        )
        for seed in (1, 19, 37)
    ]
    assert len(set(outputs)) == 1
