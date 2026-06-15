"""Unit tests for bashgym.research.report.render_report — pure, no I/O."""
from datetime import datetime, timedelta

from bashgym.research.report import render_report
from bashgym.research.scoring import DatasetMetadata, ScoredDataset


def _scored(repo_id: str, score: float, **meta_overrides) -> ScoredDataset:
    meta = DatasetMetadata(
        repo_id=repo_id,
        tags=["code-generation"],
        license="apache-2.0",
        num_rows=1000,
        features={"messages": "list"},
        last_modified=datetime.now() - timedelta(days=10),
        downloads=5000,
    )
    for k, v in meta_overrides.items():
        setattr(meta, k, v)
    return ScoredDataset(
        repo_id=repo_id,
        score=score,
        reasons=[f"+{score:.2f} task match: mocked"],
        warnings=[],
        bashgym_format="sft",
        download_command=f"DataDesignerPipeline(PipelineConfig(pipeline='coding_agent_sft', num_records=1000)).from_dataset(source='{repo_id}', split='train', column_mapping=None)",
        metadata=meta,
    )


class TestRenderReport:
    def test_empty_input_produces_header_only(self):
        md = render_report(accepted=[], rejected=[])
        assert "# HuggingFace Dataset Research Report" in md
        assert "No candidates" in md or "Top 20" in md

    def test_header_contains_generation_date(self):
        md = render_report(accepted=[_scored("a/b", 5.0)], rejected=[])
        assert "Generated" in md

    def test_top_20_table_limited(self):
        many = [_scored(f"org/ds{i}", float(i % 10)) for i in range(30)]
        md = render_report(accepted=many, rejected=[])
        table_section = md.split("## Details")[0] if "## Details" in md else md
        row_count = table_section.count("\n| `org/")
        assert row_count <= 20

    def test_accepted_section_contains_download_command(self):
        s = _scored("HuggingFaceH4/ultrachat_200k", 7.5)
        md = render_report(accepted=[s], rejected=[])
        assert "HuggingFaceH4/ultrachat_200k" in md
        assert "DataDesignerPipeline" in md

    def test_rejected_section_lists_reasons(self):
        rejected_ds = _scored("bad/dataset", 0.0)
        rejected_ds.rejected = True
        rejected_ds.rejection_reason = "license 'cc-by-nc-4.0' is non-commercial"
        md = render_report(accepted=[], rejected=[rejected_ds])
        assert "Rejected" in md
        assert "non-commercial" in md
        assert "bad/dataset" in md

    def test_ordering_by_score_desc(self):
        a = _scored("org/low", 2.0)
        b = _scored("org/high", 9.0)
        c = _scored("org/mid", 5.0)
        md = render_report(accepted=[a, b, c], rejected=[])
        i_high = md.find("org/high")
        i_mid = md.find("org/mid")
        i_low = md.find("org/low")
        assert 0 < i_high < i_mid < i_low
