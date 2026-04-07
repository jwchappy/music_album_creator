"""Shared pytest fixtures: synthetic audio files + pre-built feature dicts."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sine(
    sr: int = 22050,
    duration: float = 10.0,
    freq: float = 440.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * freq * t)
    # Add mild harmonics for a realistic spectral centroid
    wave += (amplitude * 0.3) * np.sin(2 * np.pi * freq * 2 * t)
    wave += (amplitude * 0.1) * np.sin(2 * np.pi * freq * 3 * t)
    return wave.astype(np.float32)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def tmp_audio_dir():
    """Temp directory containing 8 synthetic MP3-named WAV files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sr = 22050
        for i in range(8):
            freq = 220 * (i + 1)          # 220 Hz … 1760 Hz
            amp  = 0.15 + i * 0.04        # increasing amplitude
            wav  = _make_sine(sr=sr, duration=15.0, freq=freq, amplitude=amp)
            # Boost intro energy for even-indexed tracks
            if i % 2 == 0:
                wav[: int(5 * sr)] *= 1.8
            path = Path(tmpdir) / f"track_{i+1:02d}.mp3"
            sf.write(str(path), wav, sr)
        yield tmpdir


@pytest.fixture(scope="session")
def sample_feat() -> dict:
    """A minimal feature dict for unit-testing scoring functions."""
    return {
        "filepath":         "/fake/track.mp3",
        "filename":         "track.mp3",
        "duration":         200.0,
        "rms_mean":         0.11,
        "rms_max":          0.25,
        "intro_rms_mean":   0.18,
        "intro_impact_raw": 1.6,
        "centroid_mean":    3700.0,
        "energy_build_raw": 0.02,
        "dynamic_range_raw": 0.14,
        "tempo":            120.0,
        "key":              "C",
        "valence":          0.55,
        "energy_level":     0.11,
    }


@pytest.fixture
def scored_list_8(sample_feat) -> list[dict]:
    """8 distinct scored track dicts for ordering tests."""
    from spotify_analyzer.scoring import compute_scores

    tracks = []
    keys = ["C", "D", "E", "F", "G", "A", "B", "C#"]
    for i in range(8):
        feat = {
            **sample_feat,
            "filename":         f"track_{i+1:02d}.mp3",
            "filepath":         f"/fake/track_{i+1:02d}.mp3",
            "tempo":            90 + i * 7,           # 90–139 BPM
            "centroid_mean":    2000 + i * 400,        # 2000–4800 Hz
            "energy_level":     0.05 + i * 0.03,      # 0.05–0.26
            "rms_mean":         0.05 + i * 0.03,
            "rms_max":          0.10 + i * 0.04,
            "intro_impact_raw": 1.0 + (i % 3) * 0.5,  # varies
            "energy_build_raw": -0.01 + i * 0.005,
            "dynamic_range_raw": 0.05 + i * 0.01,
            "duration":         180 + i * 15,
            "key":              keys[i],
            "valence":          0.3 + i * 0.05,
        }
        tracks.append(compute_scores(feat))
    return tracks
