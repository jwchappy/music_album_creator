"""Integration tests for spotify_analyzer.features (requires real audio I/O)."""

import pytest
from spotify_analyzer.features import extract_features, KEY_NAMES


class TestExtractFeatures:
    """Tests that write a real WAV, then extract features from it."""

    def test_returns_expected_keys(self, tmp_audio_dir):
        import os
        from pathlib import Path

        mp3s = sorted(Path(tmp_audio_dir).glob("*.mp3"))
        assert mp3s, "fixture produced no audio files"
        result = extract_features(str(mp3s[0]))

        expected_keys = {
            "filepath", "filename", "duration", "rms_mean", "rms_max",
            "intro_rms_mean", "intro_impact_raw", "centroid_mean",
            "energy_build_raw", "dynamic_range_raw", "tempo", "key",
            "valence", "energy_level",
        }
        assert expected_keys.issubset(result.keys())

    def test_duration_positive(self, tmp_audio_dir):
        from pathlib import Path
        mp3 = next(Path(tmp_audio_dir).glob("*.mp3"))
        result = extract_features(str(mp3))
        assert result["duration"] > 0

    def test_rms_mean_positive(self, tmp_audio_dir):
        from pathlib import Path
        mp3 = next(Path(tmp_audio_dir).glob("*.mp3"))
        result = extract_features(str(mp3))
        assert result["rms_mean"] > 0

    def test_key_is_valid(self, tmp_audio_dir):
        from pathlib import Path
        mp3 = next(Path(tmp_audio_dir).glob("*.mp3"))
        result = extract_features(str(mp3))
        assert result["key"] in KEY_NAMES

    def test_valence_in_range(self, tmp_audio_dir):
        from pathlib import Path
        mp3 = next(Path(tmp_audio_dir).glob("*.mp3"))
        result = extract_features(str(mp3))
        assert 0.0 <= result["valence"] <= 1.0

    def test_intro_impact_raw_positive(self, tmp_audio_dir):
        from pathlib import Path
        mp3 = next(Path(tmp_audio_dir).glob("*.mp3"))
        result = extract_features(str(mp3))
        assert result["intro_impact_raw"] > 0

    def test_centroid_mean_in_realistic_range(self, tmp_audio_dir):
        from pathlib import Path
        mp3 = next(Path(tmp_audio_dir).glob("*.mp3"))
        result = extract_features(str(mp3))
        # For typical audio the centroid is between 100 Hz and 12 kHz
        assert 100 <= result["centroid_mean"] <= 12000

    def test_energy_level_equals_rms_mean(self, tmp_audio_dir):
        from pathlib import Path
        mp3 = next(Path(tmp_audio_dir).glob("*.mp3"))
        result = extract_features(str(mp3))
        assert result["energy_level"] == pytest.approx(result["rms_mean"])
