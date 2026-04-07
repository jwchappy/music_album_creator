"""Audio feature extraction from MP3/audio files via librosa."""

import warnings
from pathlib import Path

import numpy as np
import librosa

warnings.filterwarnings("ignore")

KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def extract_features(filepath: str) -> dict:
    """Extract all audio features from an audio file.

    Returns a dict with raw feature values used by both rank and order modes.
    """
    y, sr = librosa.load(filepath, sr=None, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # RMS energy (frame-level)
    rms = librosa.feature.rms(y=y)[0]
    rms_mean = float(np.mean(rms))
    rms_max = float(np.max(rms))

    # Intro impact: RMS of first 5 s vs. track average
    intro_samples = min(int(5 * sr), len(y))
    rms_intro = librosa.feature.rms(y=y[:intro_samples])[0]
    intro_rms_mean = float(np.mean(rms_intro))
    intro_impact_raw = intro_rms_mean / (rms_mean + 1e-9)

    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    centroid_mean = float(np.mean(spectral_centroid))

    # Energy build: first half vs. second half RMS
    mid = len(y) // 2
    rms_first = float(np.mean(librosa.feature.rms(y=y[:mid])[0]))
    rms_second = float(np.mean(librosa.feature.rms(y=y[mid:])[0]))
    energy_build_raw = rms_second - rms_first

    # Dynamic range: max − mean RMS
    dynamic_range_raw = rms_max - rms_mean

    # Tempo (BPM) — atleast_1d handles both scalar and array librosa returns
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # Chromagram-based tonal center estimate
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_idx = int(np.argmax(chroma_mean))
    key_name = KEY_NAMES[key_idx]

    # Valence proxy: blend of spectral brightness and tempo normalised to [0,1]
    tempo_norm = min(tempo / 200.0, 1.0)
    centroid_norm = min(centroid_mean / 8000.0, 1.0)
    valence = (centroid_norm + tempo_norm) / 2.0

    return {
        "filepath": filepath,
        "filename": Path(filepath).name,
        "duration": duration,
        "rms_mean": rms_mean,
        "rms_max": rms_max,
        "intro_rms_mean": intro_rms_mean,
        "intro_impact_raw": intro_impact_raw,
        "centroid_mean": centroid_mean,
        "energy_build_raw": energy_build_raw,
        "dynamic_range_raw": dynamic_range_raw,
        "tempo": tempo,
        "key": key_name,
        "valence": valence,
        "energy_level": rms_mean,  # energy_level is the normalised RMS mean
    }
