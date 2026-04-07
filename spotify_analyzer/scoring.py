"""Scoring functions for Spotify performance potential."""

import numpy as np


# ---------------------------------------------------------------------------
# Normalisation helper
# ---------------------------------------------------------------------------

def normalize(value: float, lo: float, hi: float) -> float:
    """Linearly clamp *value* to [lo, hi] and rescale to [0, 100]."""
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, (value - lo) / (hi - lo) * 100.0))


# ---------------------------------------------------------------------------
# Individual metric scores (all return 0–100)
# ---------------------------------------------------------------------------

def score_intro_impact(raw: float) -> float:
    """raw = intro_rms / track_rms_mean.  Ideal ≥ 1.5."""
    return normalize(raw, 0.5, 2.5)


def score_spectral_brightness(centroid_hz: float) -> float:
    """Target band 3 300–4 100 Hz → 100; falls off symmetrically outside."""
    if 3300 <= centroid_hz <= 4100:
        return 100.0
    if centroid_hz < 3300:
        return normalize(centroid_hz, 500, 3300)
    return normalize(8000 - centroid_hz, 0, 8000 - 4100)


def score_energy_build(raw: float, rms_mean: float) -> float:
    """Positive second-half build → higher score."""
    relative = raw / (rms_mean + 1e-9)
    return normalize(relative, -1.0, 1.0)


def score_dynamic_range(raw: float) -> float:
    """Higher dynamic range is better; capped at 0.15."""
    return normalize(raw, 0.0, 0.15)


def score_tempo(bpm: float) -> float:
    """Ideal K-pop/indie-pop: 95–135 BPM → 100; degrades linearly outside."""
    if 95 <= bpm <= 135:
        return 100.0
    if bpm < 95:
        return normalize(bpm, 40, 95)
    return normalize(200 - bpm, 0, 200 - 135)


def score_duration(seconds: float) -> float:
    """Penalise tracks longer than 4:00 (240 s) for Spotify completion rate."""
    if seconds <= 240:
        return 100.0
    return normalize(360 - seconds, 0, 360 - 240)


def score_loudness(rms_mean: float) -> float:
    """RMS proxy for ~−14 LUFS.  Peak score at rms_mean ≈ 0.11."""
    target = 0.11
    deviation = abs(rms_mean - target)
    return normalize(0.15 - deviation, 0.0, 0.15)


# ---------------------------------------------------------------------------
# Weighted composite score
# ---------------------------------------------------------------------------

WEIGHTS = {
    "intro":    0.25,
    "bright":   0.20,
    "build":    0.15,
    "dynamic":  0.10,
    "tempo":    0.15,
    "duration": 0.10,
    "loudness": 0.05,
}


def compute_scores(feat: dict) -> dict:
    """Add per-metric scores and weighted total to a feature dict."""
    s_intro  = score_intro_impact(feat["intro_impact_raw"])
    s_bright = score_spectral_brightness(feat["centroid_mean"])
    s_build  = score_energy_build(feat["energy_build_raw"], feat["rms_mean"])
    s_dyn    = score_dynamic_range(feat["dynamic_range_raw"])
    s_tempo  = score_tempo(feat["tempo"])
    s_dur    = score_duration(feat["duration"])
    s_loud   = score_loudness(feat["rms_mean"])

    total = (
        s_intro  * WEIGHTS["intro"]
        + s_bright * WEIGHTS["bright"]
        + s_build  * WEIGHTS["build"]
        + s_dyn    * WEIGHTS["dynamic"]
        + s_tempo  * WEIGHTS["tempo"]
        + s_dur    * WEIGHTS["duration"]
        + s_loud   * WEIGHTS["loudness"]
    )

    return {
        **feat,
        "s_intro":  s_intro,
        "s_bright": s_bright,
        "s_build":  s_build,
        "s_dyn":    s_dyn,
        "s_tempo":  s_tempo,
        "s_dur":    s_dur,
        "s_loud":   s_loud,
        "total":    total,
    }


def strengths_warnings(scored: dict) -> tuple[list[str], list[str]]:
    """Return (strengths, warnings) label lists based on sub-scores."""
    strengths, warnings = [], []

    if scored["s_intro"] >= 70:
        strengths.append("Strong intro")
    elif scored["s_intro"] < 40:
        warnings.append("Weak intro")

    if scored["s_bright"] >= 70:
        strengths.append("Bright mix")
    elif scored["s_bright"] < 40:
        warnings.append("Dull spectral")

    if scored["s_build"] >= 70:
        strengths.append("Good energy build")
    elif scored["s_build"] < 30:
        warnings.append("Energy drops in 2nd half")

    if scored["s_tempo"] >= 80:
        strengths.append("Ideal BPM")
    elif scored["s_tempo"] < 40:
        warnings.append("BPM out of range")

    if scored["duration"] > 240:
        warnings.append(f"Long track ({scored['duration']:.0f}s)")

    if scored["s_loud"] < 40:
        warnings.append("Loudness off-target")

    return strengths, warnings
