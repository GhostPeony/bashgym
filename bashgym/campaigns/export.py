"""Deterministic, hash-reconciled campaign evidence exports."""

from __future__ import annotations

import csv
import hashlib
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bashgym.campaigns.contracts import canonical_hash
from bashgym.campaigns.reporting import (
    write_campaign_docx,
    write_campaign_pdf,
    write_loss_png,
)


class CampaignExportError(ValueError):
    """Stable export failure for invalid or unsafe evidence projections."""


@dataclass(frozen=True)
class CampaignExportSnapshot:
    campaign: dict[str, Any]
    attempts: tuple[dict[str, Any], ...] = ()
    artifacts: tuple[dict[str, Any], ...] = ()
    comparisons: tuple[dict[str, Any], ...] = ()
    loss_by_attempt: dict[str, tuple[dict[str, Any], ...]] | None = None
    flags: tuple[str, ...] = ()

    def safe_payload(self) -> dict[str, Any]:
        artifacts = []
        for item in self.artifacts:
            if "uri" in item or "path" in item or "sealed_result_uri" in item:
                raise CampaignExportError("campaign_export_contains_local_path")
            artifacts.append(
                {
                    key: item[key]
                    for key in (
                        "artifact_id",
                        "producer_action_id",
                        "sha256",
                        "size_bytes",
                        "schema_name",
                        "sealed",
                        "valid",
                        "metadata",
                        "created_at",
                    )
                    if key in item
                }
            )
        attempts = []
        for item in self.attempts:
            if "sealed_result_uri" in item or "uri" in item or "path" in item:
                raise CampaignExportError("campaign_export_contains_local_path")
            attempts.append(dict(item))
        return {
            "schema_version": "campaign_export_snapshot.v1",
            "campaign": self.campaign,
            "attempts": attempts,
            "artifacts": artifacts,
            "comparisons": list(self.comparisons),
            "loss_by_attempt": {
                key: list(value)
                for key, value in sorted((self.loss_by_attempt or {}).items())
            },
            "flags": list(self.flags),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                    for key, value in row.items()
                }
            )


def _loss_svg(snapshot: dict[str, Any]) -> str:
    width, height = 960, 480
    left, right, top, bottom = 70, 30, 45, 65
    series = []
    attempt_by_id = {
        item.get("attempt_id"): item for item in snapshot["attempts"] if item.get("attempt_id")
    }
    for attempt_id, values in snapshot["loss_by_attempt"].items():
        points = [
            (int(item["step"]), float(item["value"]))
            for item in values
            if "step" in item and "value" in item
        ]
        if points:
            series.append((attempt_id, attempt_by_id.get(attempt_id, {}).get("stage", "unknown"), points))
    max_step = max((step for _id, _stage, points in series for step, _value in points), default=1)
    losses = [value for _id, _stage, points in series for _step, value in points]
    min_loss = min(losses, default=0.0)
    max_loss = max(losses, default=1.0)
    if max_loss == min_loss:
        max_loss = min_loss + 1.0

    def x(step: int) -> float:
        return left + (width - left - right) * (step / max_step)

    def y(loss: float) -> float:
        return top + (height - top - bottom) * ((max_loss - loss) / (max_loss - min_loss))

    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="960" height="480" viewBox="0 0 960 480">',
        '<rect width="960" height="480" fill="#fffdf7"/>',
        '<text x="70" y="28" font-family="sans-serif" font-size="18" font-weight="700">Campaign training loss</text>',
        f'<line x1="{left}" y1="{height-bottom}" x2="{width-right}" y2="{height-bottom}" stroke="#332f2a" stroke-width="2"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{height-bottom}" stroke="#332f2a" stroke-width="2"/>',
        f'<text x="{width/2:.0f}" y="455" text-anchor="middle" font-family="sans-serif" font-size="13">Training step</text>',
        f'<text x="18" y="{height/2:.0f}" text-anchor="middle" transform="rotate(-90 18 {height/2:.0f})" font-family="sans-serif" font-size="13">Loss</text>',
    ]
    colors = ("#7c3aed", "#d97706", "#15803d", "#0369a1", "#be123c")
    for index, (attempt_id, stage, points) in enumerate(series):
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{x(step):.2f},{y(value):.2f}" for step, value in points)
        dash = ' stroke-dasharray="7 5"' if stage == "smoke_training" else ""
        lines.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="3"{dash}/>'
        )
        label = html.escape(f"{attempt_id} · {stage}")
        lines.append(
            f'<text x="{left + 10}" y="{top + 20 + index * 18}" font-family="monospace" font-size="11" fill="{color}">{label}</text>'
        )
    if not series:
        lines.append(
            '<text x="480" y="240" text-anchor="middle" font-family="sans-serif" font-size="16" fill="#6b645d">No persisted loss series</text>'
        )
    lines.append(
        '<text x="930" y="455" text-anchor="end" font-family="sans-serif" font-size="10" fill="#6b645d">Dashed = smoke engineering evidence</text>'
    )
    lines.append("</svg>\n")
    return "\n".join(lines)


def _markdown(snapshot: dict[str, Any], source_digest: str) -> str:
    campaign = snapshot["campaign"]
    attempts = snapshot["attempts"]
    comparisons = snapshot["comparisons"]
    smoke = [item for item in attempts if item.get("stage") == "smoke_training"]
    full = [item for item in attempts if item.get("stage") == "full_training"]
    completed_full = [item for item in full if item.get("status") == "completed"]
    quality_ready = bool(completed_full and comparisons)
    lines = [
        "# Campaign Evidence Report",
        "",
        f"- Campaign: `{campaign.get('campaign_id', 'unknown')}`",
        f"- Objective: {campaign.get('objective', 'Not recorded')}",
        f"- Status: `{campaign.get('status', 'unknown')}`",
        f"- Champion: `{campaign.get('champion_ref') or 'unchanged / not recorded'}`",
        f"- Evidence digest: `{source_digest}`",
        "",
        "## Model-quality findings",
        "",
    ]
    if quality_ready:
        latest = comparisons[-1]
        lines.extend(
            [
                f"The latest deterministic development gate verdict is **{latest.get('verdict', 'unknown')}**.",
                "",
                "This section is backed by at least one completed full-training attempt and a persisted comparison.",
            ]
        )
    else:
        lines.append(
            "No model-quality findings are claimed. A completed full-training attempt and persisted comparison are both required."
        )
    lines.extend(["", "## Engineering evidence", ""])
    lines.append(f"- Smoke attempts: {len(smoke)} (runtime/semantics/memory evidence only)")
    lines.append(f"- Full-training attempts: {len(full)}")
    lines.append(f"- Persisted comparison records: {len(comparisons)}")
    lines.extend(["", "## Attempts", "", "| Attempt | Stage | Status | Candidate digest |", "|---|---|---|---|"])
    for item in attempts:
        digest = str(item.get("candidate_digest", ""))
        lines.append(
            f"| `{item.get('attempt_id', '')}` | {item.get('stage', '')} | {item.get('status', '')} | `{digest[:12]}` |"
        )
    lines.extend(["", "## Sealed evidence", "", "| Schema | SHA-256 | Bytes | Valid |", "|---|---|---:|---|"])
    for item in snapshot["artifacts"]:
        lines.append(
            f"| {item.get('schema_name', '')} | `{item.get('sha256', '')}` | {item.get('size_bytes', 0)} | {item.get('valid', False)} |"
        )
    lines.extend(["", "## Flags", ""])
    if snapshot["flags"]:
        lines.extend(f"- {flag}" for flag in snapshot["flags"])
    else:
        lines.append("- No implementation flags were recorded for this export.")
    lines.extend(
        [
            "",
            "## Reconciliation",
            "",
            "Every table and chart in this package is derived from `campaign_evidence.json`; `export_manifest.json` records the SHA-256 of each generated file.",
            "",
        ]
    )
    return "\n".join(lines)


def export_campaign_evidence(
    value: CampaignExportSnapshot,
    output_directory: Path,
) -> dict[str, Any]:
    """Write deterministic evidence, chart, Word, and PDF projections."""

    snapshot = value.safe_payload()
    source_digest = canonical_hash(snapshot)
    output_directory.mkdir(parents=True, exist_ok=True)
    if any(output_directory.iterdir()):
        raise CampaignExportError("campaign_export_directory_not_empty")

    evidence_path = output_directory / "campaign_evidence.json"
    evidence_path.write_bytes(_json_bytes(snapshot) + b"\n")
    _write_csv(
        output_directory / "attempts.csv",
        ("attempt_id", "study_id", "stage", "status", "candidate_digest", "created_at", "updated_at"),
        snapshot["attempts"],
    )
    _write_csv(
        output_directory / "artifacts.csv",
        ("artifact_id", "producer_action_id", "schema_name", "sha256", "size_bytes", "sealed", "valid", "created_at"),
        snapshot["artifacts"],
    )
    _write_csv(
        output_directory / "comparisons.csv",
        ("comparison_digest", "champion_digest", "candidate_digest", "sample_count", "verdict", "blocking_reasons", "warnings", "created_at"),
        snapshot["comparisons"],
    )
    (output_directory / "training_loss.svg").write_text(_loss_svg(snapshot), encoding="utf-8")
    loss_png = output_directory / "training_loss.png"
    write_loss_png(snapshot, loss_png)
    (output_directory / "campaign_report.md").write_text(
        _markdown(snapshot, source_digest), encoding="utf-8", newline="\n"
    )
    write_campaign_docx(
        snapshot,
        source_digest,
        loss_png,
        output_directory / "campaign_report.docx",
    )
    write_campaign_pdf(
        snapshot,
        source_digest,
        loss_png,
        output_directory / "campaign_report.pdf",
    )

    files = []
    for path in sorted(output_directory.iterdir(), key=lambda item: item.name):
        files.append({"name": path.name, "sha256": _sha256(path), "size_bytes": path.stat().st_size})
    manifest = {
        "schema_version": "campaign_export_manifest.v1",
        "campaign_id": snapshot["campaign"].get("campaign_id"),
        "source_digest": source_digest,
        "quality_findings_available": bool(
            any(
                item.get("stage") == "full_training" and item.get("status") == "completed"
                for item in snapshot["attempts"]
            )
            and snapshot["comparisons"]
        ),
        "files": files,
    }
    (output_directory / "export_manifest.json").write_bytes(_json_bytes(manifest) + b"\n")
    return manifest


__all__ = [
    "CampaignExportError",
    "CampaignExportSnapshot",
    "export_campaign_evidence",
]
