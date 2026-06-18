"""Tests for benchmark decontamination."""

from bashgym.datasets.decontaminate import Decontaminator, ngrams

BENCH_LONG = "the quick brown fox jumps over the lazy dog near the old red barn today"
BENCH_SHORT = "install the package and run the tests"


class TestNgrams:
    def test_basic(self):
        assert ngrams(["a", "b", "c"], 2) == {("a", "b"), ("b", "c")}

    def test_shorter_than_n_returns_whole(self):
        assert ngrams(["a", "b"], 3) == {("a", "b")}


class TestDecontaminator:
    def test_clean_example_kept(self):
        d = Decontaminator([BENCH_LONG])
        assert d.contamination_reason("a totally unrelated sentence about cats and weather") is None

    def test_shared_13gram_dropped(self):
        d = Decontaminator([BENCH_LONG])
        leaked = "yesterday i saw the quick brown fox jumps over the lazy dog near the old red barn"
        assert d.contamination_reason(leaked) == "13gram_overlap"

    def test_high_3gram_jaccard_dropped(self):
        d = Decontaminator([BENCH_SHORT])
        near = "install the package and run the tests now please"
        assert d.contamination_reason(near) == "3gram_jaccard"

    def test_filter_reports_counts(self):
        d = Decontaminator([BENCH_LONG])
        examples = [
            {"t": "clean unrelated content here"},
            {"t": "x the quick brown fox jumps over the lazy dog near the old red barn z"},
        ]
        kept, report = d.filter(examples, text_of=lambda e: e["t"])
        assert report.kept == 1 and report.dropped == 1
        assert report.drop_reasons.get("13gram_overlap") == 1
        assert kept[0]["t"].startswith("clean")
