"""Tests for the self/public dataset mixer."""

import pytest

from bashgym.datasets.mixer import mix_datasets


class TestMixDatasets:
    def test_hits_target_fraction_by_upsampling_small_self(self):
        self_ex = [{"s": i} for i in range(10)]
        public = [{"p": i} for i in range(80)]
        mixed, report = mix_datasets(self_ex, public, self_fraction=0.2, seed=0)
        assert report.self_fraction == pytest.approx(0.2, abs=0.02)
        assert len(mixed) == report.n_public_out + report.n_self_out

    def test_samples_down_when_self_is_large(self):
        self_ex = [{"s": i} for i in range(200)]
        public = [{"p": i} for i in range(80)]
        mixed, report = mix_datasets(self_ex, public, self_fraction=0.2, seed=0)
        # target = 0.2/0.8*80 = 20; self had plenty, so it samples 20.
        assert report.n_self_out == 20
        assert report.self_fraction == pytest.approx(0.2, abs=0.02)

    def test_no_public_returns_self(self):
        self_ex = [{"s": 1}, {"s": 2}]
        mixed, report = mix_datasets(self_ex, [], self_fraction=0.2)
        assert mixed == self_ex and report.self_fraction == 1.0

    def test_no_upsample_keeps_self_as_is(self):
        self_ex = [{"s": i} for i in range(10)]
        public = [{"p": i} for i in range(80)]
        _, report = mix_datasets(self_ex, public, self_fraction=0.2, upsample=False)
        assert report.n_self_out == 10  # not upsampled to 20

    def test_invalid_fraction_raises(self):
        with pytest.raises(ValueError):
            mix_datasets([{"s": 1}], [{"p": 1}], self_fraction=1.5)
