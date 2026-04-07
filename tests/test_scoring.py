"""Unit tests for spotify_analyzer.scoring."""

import pytest
from spotify_analyzer.scoring import (
    normalize,
    score_intro_impact,
    score_spectral_brightness,
    score_energy_build,
    score_dynamic_range,
    score_tempo,
    score_duration,
    score_loudness,
    compute_scores,
    strengths_warnings,
    WEIGHTS,
)


# ---------------------------------------------------------------------------
# normalize
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_midpoint(self):
        assert normalize(5.0, 0.0, 10.0) == pytest.approx(50.0)

    def test_at_lo_returns_zero(self):
        assert normalize(0.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_at_hi_returns_hundred(self):
        assert normalize(10.0, 0.0, 10.0) == pytest.approx(100.0)

    def test_below_lo_clamped(self):
        assert normalize(-5.0, 0.0, 10.0) == pytest.approx(0.0)

    def test_above_hi_clamped(self):
        assert normalize(15.0, 0.0, 10.0) == pytest.approx(100.0)

    def test_degenerate_lo_eq_hi_returns_50(self):
        assert normalize(7.0, 7.0, 7.0) == pytest.approx(50.0)


# ---------------------------------------------------------------------------
# Intro impact
# ---------------------------------------------------------------------------

class TestScoreIntroImpact:
    def test_ideal_raw_scores_high(self):
        assert score_intro_impact(2.0) >= 70.0

    def test_low_raw_scores_low(self):
        assert score_intro_impact(0.5) == pytest.approx(0.0)

    def test_high_raw_clamped_to_100(self):
        assert score_intro_impact(10.0) == pytest.approx(100.0)

    def test_returns_float(self):
        assert isinstance(score_intro_impact(1.5), float)


# ---------------------------------------------------------------------------
# Spectral brightness
# ---------------------------------------------------------------------------

class TestScoreSpectralBrightness:
    def test_target_band_returns_100(self):
        for hz in (3300, 3700, 4100):
            assert score_spectral_brightness(hz) == pytest.approx(100.0), hz

    def test_below_band_lower_score(self):
        assert score_spectral_brightness(1000) < 50.0

    def test_above_band_lower_score(self):
        # 7 500 Hz is far above the 4 100 Hz ceiling → clearly below 50
        assert score_spectral_brightness(7500) < 50.0

    def test_very_low_hz_returns_zero(self):
        assert score_spectral_brightness(100) == pytest.approx(0.0)

    def test_monotone_increase_below_band(self):
        scores = [score_spectral_brightness(hz) for hz in (500, 1500, 2500, 3300)]
        assert scores == sorted(scores)


# ---------------------------------------------------------------------------
# Energy build
# ---------------------------------------------------------------------------

class TestScoreEnergyBuild:
    def test_positive_build_scores_high(self):
        assert score_energy_build(0.05, 0.10) > 50.0

    def test_negative_build_scores_low(self):
        assert score_energy_build(-0.05, 0.10) < 50.0

    def test_zero_build_near_midpoint(self):
        s = score_energy_build(0.0, 0.10)
        assert 40.0 <= s <= 60.0

    def test_tiny_rms_no_division_error(self):
        # Should not raise ZeroDivisionError
        score_energy_build(0.01, 0.0)


# ---------------------------------------------------------------------------
# Dynamic range
# ---------------------------------------------------------------------------

class TestScoreDynamicRange:
    def test_zero_range_returns_zero(self):
        assert score_dynamic_range(0.0) == pytest.approx(0.0)

    def test_cap_returns_100(self):
        assert score_dynamic_range(0.15) == pytest.approx(100.0)

    def test_above_cap_clamped(self):
        assert score_dynamic_range(0.5) == pytest.approx(100.0)

    def test_intermediate_value(self):
        s = score_dynamic_range(0.075)
        assert 45.0 <= s <= 55.0


# ---------------------------------------------------------------------------
# Tempo
# ---------------------------------------------------------------------------

class TestScoreTempo:
    @pytest.mark.parametrize("bpm", [95, 110, 120, 135])
    def test_ideal_range_returns_100(self, bpm):
        assert score_tempo(bpm) == pytest.approx(100.0)

    def test_very_low_bpm_scores_zero(self):
        assert score_tempo(0) == pytest.approx(0.0)

    def test_very_high_bpm_scores_zero(self):
        assert score_tempo(250) == pytest.approx(0.0)

    def test_moderate_low_bpm(self):
        assert 0.0 < score_tempo(70) < 100.0

    def test_moderate_high_bpm(self):
        assert 0.0 < score_tempo(160) < 100.0


# ---------------------------------------------------------------------------
# Duration
# ---------------------------------------------------------------------------

class TestScoreDuration:
    def test_under_240s_full_score(self):
        for s in (60, 180, 239, 240):
            assert score_duration(s) == pytest.approx(100.0), s

    def test_over_360s_zero(self):
        assert score_duration(360) == pytest.approx(0.0)

    def test_very_long_clamped_zero(self):
        assert score_duration(600) == pytest.approx(0.0)

    def test_midpoint_between_240_360(self):
        s = score_duration(300)
        assert 45.0 <= s <= 55.0


# ---------------------------------------------------------------------------
# Loudness
# ---------------------------------------------------------------------------

class TestScoreLoudness:
    def test_target_rms_scores_high(self):
        assert score_loudness(0.11) > 90.0

    def test_far_from_target_scores_low(self):
        assert score_loudness(0.60) < 20.0

    def test_zero_rms_not_error(self):
        assert 0.0 <= score_loudness(0.0) <= 100.0


# ---------------------------------------------------------------------------
# compute_scores
# ---------------------------------------------------------------------------

class TestComputeScores:
    def test_adds_score_keys(self, sample_feat):
        result = compute_scores(sample_feat)
        for key in ("s_intro", "s_bright", "s_build", "s_dyn",
                    "s_tempo", "s_dur", "s_loud", "total"):
            assert key in result, key

    def test_total_range(self, sample_feat):
        result = compute_scores(sample_feat)
        assert 0.0 <= result["total"] <= 100.0

    def test_weights_sum_to_one(self):
        assert sum(WEIGHTS.values()) == pytest.approx(1.0)

    def test_ideal_track_scores_high(self):
        ideal = {
            "filepath": "/fake/ideal.mp3",
            "filename": "ideal.mp3",
            "duration": 200.0,
            "rms_mean": 0.11,
            "rms_max": 0.24,
            "intro_rms_mean": 0.22,
            "intro_impact_raw": 2.0,
            "centroid_mean": 3700.0,
            "energy_build_raw": 0.03,
            "dynamic_range_raw": 0.13,
            "tempo": 120.0,
            "key": "C",
            "valence": 0.6,
            "energy_level": 0.11,
        }
        result = compute_scores(ideal)
        assert result["total"] >= 70.0

    def test_poor_track_scores_low(self):
        poor = {
            "filepath": "/fake/poor.mp3",
            "filename": "poor.mp3",
            "duration": 400.0,
            "rms_mean": 0.005,
            "rms_max": 0.006,
            "intro_rms_mean": 0.003,
            "intro_impact_raw": 0.6,
            "centroid_mean": 300.0,
            "energy_build_raw": -0.002,
            "dynamic_range_raw": 0.001,
            "tempo": 20.0,
            "key": "C",
            "valence": 0.1,
            "energy_level": 0.005,
        }
        result = compute_scores(poor)
        assert result["total"] <= 30.0


# ---------------------------------------------------------------------------
# strengths_warnings
# ---------------------------------------------------------------------------

class TestStrengthsWarnings:
    def test_high_score_track_has_strengths(self, sample_feat):
        # sample_feat is already near-ideal
        scored = compute_scores(sample_feat)
        strengths, _ = strengths_warnings(scored)
        assert len(strengths) >= 1

    def test_long_track_triggers_warning(self, sample_feat):
        feat = {**sample_feat, "duration": 350.0}
        scored = compute_scores(feat)
        _, warns = strengths_warnings(scored)
        assert any("Long" in w for w in warns)

    def test_out_of_range_bpm_triggers_warning(self, sample_feat):
        feat = {**sample_feat, "tempo": 20.0}
        scored = compute_scores(feat)
        _, warns = strengths_warnings(scored)
        assert any("BPM" in w for w in warns)

    def test_returns_lists(self, sample_feat):
        scored = compute_scores(sample_feat)
        s, w = strengths_warnings(scored)
        assert isinstance(s, list)
        assert isinstance(w, list)
