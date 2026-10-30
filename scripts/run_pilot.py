"""Run exactly one budget-checked article and preserve its full result for review."""

import argparse
import json
import logging
import random
import time
from pathlib import Path

from pydantic import TypeAdapter

from article_rectifier.config import Settings
from article_rectifier.pipeline import PipelineResult
from rectification_system import build_pipeline, check_provider_budget
from rectifier import rectify_article


def main() -> None:
    """Select one article, verify balance, run it, and persist evaluation inputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--article-id")
    selection.add_argument("--seed", type=int)
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    settings = Settings(RECTIFIER_DEBUG=True, RECTIFIER_CACHE_ONLY=False)
    before = check_provider_budget(settings)
    if before <= settings.minimum_budget_reserve_usd:
        raise RuntimeError("Insufficient verified provider balance")
    pool = [i for i in range(11, 105) if i not in args.exclude]
    article_id = (
        args.article_id or f"article_{random.Random(args.seed).choice(pool):03d}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(
            "Use a new output directory; preserve the previous result"
        )
    (args.output_dir / "selected-id.txt").write_text(article_id + "\n")
    pipeline = build_pipeline(settings)
    started = time.monotonic()
    result = rectify_article(article_id, pipeline, args.output_dir)
    elapsed = time.monotonic() - started
    result_path.write_bytes(TypeAdapter(PipelineResult).dump_json(result, indent=2))
    metadata = {
        "article_id": article_id,
        "seed": args.seed,
        "elapsed_seconds": elapsed,
        "balance_before_usd": before,
        "run_id": pipeline.workflow.llm_client.trace_context["run_id"],
    }
    try:
        metadata["balance_after_usd"] = check_provider_budget(settings)
    finally:
        (args.output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    print(
        json.dumps(
            {
                **metadata,
                "status": result.status,
                "accepted": len(result.accepted_edits),
                "rejected": len(result.rejected_patches),
                "unresolved_requests": len(result.unresolved_issues),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
