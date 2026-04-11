"""Markdown report generator for scored HF datasets. Pure — input list, output string."""
from __future__ import annotations

from datetime import datetime

from bashgym.research.scoring import ScoredDataset

TOP_N = 20


def render_report(accepted: list[ScoredDataset], rejected: list[ScoredDataset]) -> str:
    """Render a markdown report from scored datasets.

    ``accepted`` is ranked by ``.score`` descending before rendering.
    ``rejected`` is grouped into a summary section by rejection reason prefix.
    """
    lines: list[str] = []
    lines.append("# HuggingFace Dataset Research Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Accepted candidates: {len(accepted)} — Rejected: {len(rejected)}")
    lines.append("")

    if not accepted:
        lines.append("No candidates passed the filters. See rejected section below.")
        lines.append("")
    else:
        sorted_accepted = sorted(accepted, key=lambda s: s.score, reverse=True)
        top = sorted_accepted[:TOP_N]

        lines.append(f"## Top {len(top)} by score")
        lines.append("")
        lines.append("| Rank | Repo | Score | Format | Rows | License | Updated |")
        lines.append("|---|---|---|---|---|---|---|")
        for rank, s in enumerate(top, start=1):
            rows_str = f"{s.metadata.num_rows:,}" if s.metadata.num_rows else "?"
            lic = s.metadata.license or "unknown"
            updated = s.metadata.last_modified.date().isoformat() if s.metadata.last_modified else "?"
            fmt = s.bashgym_format or "—"
            lines.append(
                f"| {rank} | `{s.repo_id}` | {s.score:.2f} | {fmt} | {rows_str} | {lic} | {updated} |"
            )
        lines.append("")

        lines.append("## Details")
        lines.append("")
        for rank, s in enumerate(top, start=1):
            lines.append(f"### {rank}. `{s.repo_id}` — {s.score:.2f}")
            lines.append("")
            if s.bashgym_format:
                lines.append(f"**Format:** {s.bashgym_format.upper()}")
                lines.append("")
            lines.append("**Score breakdown:**")
            for r in s.reasons:
                lines.append(f"- {r}")
            if s.warnings:
                lines.append("")
                lines.append("**Warnings:**")
                for w in s.warnings:
                    lines.append(f"- {w}")
            lines.append("")
            lines.append("**Download command:**")
            lines.append("")
            lines.append("```python")
            lines.append(s.download_command)
            lines.append("```")
            lines.append("")

    if rejected:
        lines.append("## Rejected candidates")
        lines.append("")
        buckets: dict[str, list[str]] = {}
        full_reasons: dict[str, str] = {}
        for r in rejected:
            reason_full = r.rejection_reason or "unknown"
            key = reason_full.split(":")[0].strip()
            buckets.setdefault(key, []).append(r.repo_id)
            full_reasons.setdefault(key, reason_full)
        for reason_key, ids in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
            lines.append(f"### {full_reasons[reason_key]} ({len(ids)} datasets)")
            for rid in ids[:20]:
                lines.append(f"- `{rid}`")
            if len(ids) > 20:
                lines.append(f"- …and {len(ids) - 20} more")
            lines.append("")

    return "\n".join(lines) + "\n"
