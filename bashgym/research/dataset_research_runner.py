"""CLI: run an empirical dataset ranking by training on each top-N candidate.

Usage:
    python -m bashgym.research.dataset_research_runner --top-n 5 --mode simulate
    python -m bashgym.research.dataset_research_runner --top-n 10 --mode real \\
        --train-steps 100 --num-records 500 \\
        --base-model unsloth/gemma-4-E4B-it

Output:
    ~/.bashgym/research/dataset_empirical_ranking.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

from bashgym.gym.autoresearch import (
    AutoResearchConfig,
    AutoResearcher,
)
from bashgym.research.dataset_search_space import (
    DatasetCandidate,
    DatasetSearchSpace,
)
from bashgym.research.hf_dataset_scanner import CACHE_PATH as SCANNER_CACHE
from bashgym.research.scoring import DatasetMetadata, score_dataset

logger = logging.getLogger("dataset_research_runner")

OUTPUT_DIR = Path(os.path.expanduser("~/.bashgym/research"))
REPORT_PATH = OUTPUT_DIR / "dataset_empirical_ranking.md"
WORK_DIR = OUTPUT_DIR / "empirical_work"


def _load_candidates(top_n: int, cache_path: Path) -> list[DatasetCandidate]:
    """Load cached ScoredDataset entries, re-score, filter to SFT, take top-N."""
    if not cache_path.exists():
        logger.error("Scanner cache not found at %s. Run hf_dataset_scanner first.", cache_path)
        return []

    raw = json.loads(cache_path.read_text())
    scored = []
    for repo_id, d in raw.items():
        lm = d.get("last_modified")
        last_modified = None
        if lm:
            try:
                last_modified = datetime.fromisoformat(lm)
            except ValueError:
                pass
        meta = DatasetMetadata(
            repo_id=repo_id,
            tags=d.get("tags") or [],
            license=d.get("license"),
            num_rows=d.get("num_rows"),
            download_size_bytes=d.get("download_size_bytes"),
            features=d.get("features") or {},
            last_modified=last_modified,
            downloads=d.get("downloads", 0),
            gated=d.get("gated", False),
            description=d.get("description", ""),
        )
        scored.append(score_dataset(meta))

    # Include SFT + GRPO + undetected-format candidates. In simulate mode all
    # are evaluated heuristically; in real mode, DataDesignerPipeline will
    # attempt ingestion and fail gracefully on incompatible formats.
    usable = [s for s in scored if not s.rejected and s.bashgym_format in ("sft", "grpo", None)]
    usable.sort(key=lambda s: s.score, reverse=True)
    top = usable[:top_n]

    return [
        DatasetCandidate(
            repo_id=s.repo_id,
            hf_score=s.score,
            bashgym_format=s.bashgym_format or "sft",
        )
        for s in top
    ]


def _build_base_trainer_config(base_model: str):
    """Build a minimal TrainerConfig. Lazy-imported to avoid pulling in torch at CLI startup."""
    from bashgym.gym.trainer import TrainerConfig

    return TrainerConfig(
        base_model=base_model,
        strategy="sft",
        max_steps=100,
        output_dir=str(WORK_DIR / "base"),
    )


def _render_ranking_report(candidates: list[DatasetCandidate], mode: str) -> str:
    lines = []
    lines.append("# Empirical Dataset Ranking")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Mode: **{mode}**")
    lines.append(f"Candidates evaluated: {len(candidates)}")
    lines.append("")

    def _key(c: DatasetCandidate):
        if c.error:
            return (2, float("inf"))
        if c.eval_loss is None:
            return (1, float("inf"))
        return (0, c.eval_loss)

    sorted_cs = sorted(candidates, key=_key)

    lines.append("## Ranking (lower eval_loss = better)")
    lines.append("")
    lines.append("| Rank | Repo | HF Score | Eval Loss | Final Loss | Status |")
    lines.append("|---|---|---|---|---|---|")
    for rank, c in enumerate(sorted_cs, start=1):
        status = "error" if c.error else ("ok" if c.eval_loss is not None else "skipped")
        eval_str = f"{c.eval_loss:.4f}" if c.eval_loss is not None else "—"
        final_str = f"{c.final_loss:.4f}" if c.final_loss is not None else "—"
        lines.append(
            f"| {rank} | `{c.repo_id}` | {c.hf_score:.2f} | {eval_str} | {final_str} | {status} |"
        )
    lines.append("")

    errors = [c for c in candidates if c.error]
    if errors:
        lines.append("## Errors")
        lines.append("")
        for c in errors:
            lines.append(f"- `{c.repo_id}`: {c.error}")
        lines.append("")

    return "\n".join(lines) + "\n"


async def _run_async(
    top_n: int,
    mode: str,
    num_records: int,
    train_steps: int,
    base_model: str,
    cache_path: Path,
) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    candidates = _load_candidates(top_n, cache_path)
    if not candidates:
        logger.error("No SFT candidates to evaluate. Aborting.")
        return 1

    logger.info("Loaded %d SFT candidates for evaluation", len(candidates))
    for i, c in enumerate(candidates, start=1):
        logger.info("  %d. %s (hf_score=%.2f)", i, c.repo_id, c.hf_score)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    base_config = _build_base_trainer_config(base_model)

    search_space = DatasetSearchSpace(
        candidates=candidates,
        base_trainer_config=base_config,
        work_dir=WORK_DIR,
        num_records=num_records,
        train_steps=train_steps,
        mode=mode,
        hf_token=os.environ.get("HF_TOKEN"),
    )

    ar_config = AutoResearchConfig(
        max_experiments=len(candidates),
        train_steps=train_steps,
        mode=mode,
        eval_metric="eval_loss",
    )

    researcher = AutoResearcher(
        config=ar_config,
        base_trainer_config=base_config,
        dataset_path=None,
        val_dataset_path=None,
        search_space=search_space,
    )

    await researcher.run_loop(dataset_path=Path("."), callback=None)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = _render_ranking_report(candidates, mode=mode)
    REPORT_PATH.write_text(report)
    logger.info("Wrote empirical ranking to %s", REPORT_PATH)

    print()
    print("Empirical dataset ranking (lower eval_loss = better):")
    ranked = sorted(
        [c for c in candidates if c.eval_loss is not None],
        key=lambda c: c.eval_loss,
    )
    for rank, c in enumerate(ranked, start=1):
        print(f"  {rank}. {c.repo_id:<60}  eval_loss={c.eval_loss:.4f}")

    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Empirically rank HF datasets by short-run training impact."
    )
    ap.add_argument(
        "--top-n", type=int, default=5, help="Number of top-scored candidates to evaluate"
    )
    ap.add_argument("--mode", choices=["simulate", "real"], default="simulate")
    ap.add_argument(
        "--num-records", type=int, default=500, help="Records to materialize per dataset"
    )
    ap.add_argument("--train-steps", type=int, default=100, help="SFT training steps per candidate")
    ap.add_argument(
        "--base-model",
        default="unsloth/gemma-4-E4B-it",
        help="Base model for SFT runs (default: unsloth/gemma-4-E4B-it)",
    )
    ap.add_argument(
        "--cache-path",
        type=Path,
        default=SCANNER_CACHE,
        help="Path to the HF scanner cache JSON",
    )
    args = ap.parse_args()

    return asyncio.run(
        _run_async(
            top_n=args.top_n,
            mode=args.mode,
            num_records=args.num_records,
            train_steps=args.train_steps,
            base_model=args.base_model,
            cache_path=args.cache_path,
        )
    )


if __name__ == "__main__":
    sys.exit(main())
