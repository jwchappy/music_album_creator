#!/usr/bin/env python3
"""
spotify_analyzer — CLI tool for Spotify track ranking and album ordering.
"""

import os
import csv
import math
import warnings
from pathlib import Path

import click
import numpy as np
import librosa
from rich.console import Console
from rich.table import Table
from rich import box

warnings.filterwarnings("ignore")

console = Console()


# ---------------------------------------------------------------------------
# Audio feature extraction
# ---------------------------------------------------------------------------

def extract_features(filepath: str) -> dict:
    """Extract all audio features from an MP3 file."""
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

    # Energy build: first half vs. second half
    mid = len(y) // 2
    rms_first = float(np.mean(librosa.feature.rms(y=y[:mid])[0]))
    rms_second = float(np.mean(librosa.feature.rms(y=y[mid:])[0]))
    energy_build_raw = rms_second - rms_first

    # Dynamic range
    dynamic_range_raw = rms_max - rms_mean

    # Tempo (BPM)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo = float(np.atleast_1d(tempo)[0])

    # Chromagram for key/mode
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key_idx = int(np.argmax(chroma_mean))
    key_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    key_name = key_names[key_idx]

    # Mood / valence proxy
    tempo_norm = min(tempo / 200.0, 1.0)
    centroid_norm = min(centroid_mean / 8000.0, 1.0)
    valence = (centroid_norm + tempo_norm) / 2.0

    # Energy level (normalized RMS mean)
    energy_level = rms_mean

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
        "energy_level": energy_level,
    }


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

def normalize(value: float, lo: float, hi: float) -> float:
    """Linearly clamp value to [lo, hi] and scale to [0, 100]."""
    if hi == lo:
        return 50.0
    return max(0.0, min(100.0, (value - lo) / (hi - lo) * 100))


def score_intro_impact(raw: float) -> float:
    """raw = intro_rms / track_rms; ideal ≥ 1.5"""
    return normalize(raw, 0.5, 2.5)


def score_spectral_brightness(centroid_hz: float) -> float:
    """Target band: 3300–4100 Hz → 100; falls off outside."""
    if 3300 <= centroid_hz <= 4100:
        return 100.0
    elif centroid_hz < 3300:
        return normalize(centroid_hz, 500, 3300)
    else:
        return normalize(8000 - centroid_hz, 0, 8000 - 4100)


def score_energy_build(raw: float, rms_mean: float) -> float:
    """Positive build is good; normalize relative to track mean."""
    relative = raw / (rms_mean + 1e-9)
    return normalize(relative, -1.0, 1.0)


def score_dynamic_range(raw: float) -> float:
    """Higher dynamic range → better; cap at 0.15."""
    return normalize(raw, 0.0, 0.15)


def score_tempo(bpm: float) -> float:
    """Ideal K-pop/indie-pop: 95–135 BPM."""
    if 95 <= bpm <= 135:
        return 100.0
    elif bpm < 95:
        return normalize(bpm, 40, 95)
    else:
        return normalize(200 - bpm, 0, 200 - 135)


def score_duration(seconds: float) -> float:
    """Penalise tracks > 240 s (4:00)."""
    if seconds <= 240:
        return 100.0
    return normalize(360 - seconds, 0, 360 - 240)


def score_loudness(rms_mean: float) -> float:
    """
    Approximate LUFS from RMS (rough linear scale).
    Target ~-14 LUFS → RMS ≈ 0.09–0.14 for typical tracks.
    Score peaks at rms_mean ≈ 0.11.
    """
    target = 0.11
    deviation = abs(rms_mean - target)
    return normalize(0.15 - deviation, 0.0, 0.15)


def compute_scores(feat: dict) -> dict:
    s_intro = score_intro_impact(feat["intro_impact_raw"])
    s_bright = score_spectral_brightness(feat["centroid_mean"])
    s_build = score_energy_build(feat["energy_build_raw"], feat["rms_mean"])
    s_dyn = score_dynamic_range(feat["dynamic_range_raw"])
    s_tempo = score_tempo(feat["tempo"])
    s_dur = score_duration(feat["duration"])
    s_loud = score_loudness(feat["rms_mean"])

    total = (
        s_intro * 0.25
        + s_bright * 0.20
        + s_build * 0.15
        + s_dyn * 0.10
        + s_tempo * 0.15
        + s_dur * 0.10
        + s_loud * 0.05
    )

    return {
        **feat,
        "s_intro": s_intro,
        "s_bright": s_bright,
        "s_build": s_build,
        "s_dyn": s_dyn,
        "s_tempo": s_tempo,
        "s_dur": s_dur,
        "s_loud": s_loud,
        "total": total,
    }


def strengths_warnings(scored: dict) -> tuple[list[str], list[str]]:
    strengths, warnings_list = [], []

    if scored["s_intro"] >= 70:
        strengths.append("Strong intro")
    elif scored["s_intro"] < 40:
        warnings_list.append("Weak intro")

    if scored["s_bright"] >= 70:
        strengths.append("Bright mix")
    elif scored["s_bright"] < 40:
        warnings_list.append("Dull spectral")

    if scored["s_build"] >= 70:
        strengths.append("Good energy build")
    elif scored["s_build"] < 30:
        warnings_list.append("Energy drops in 2nd half")

    if scored["s_tempo"] >= 80:
        strengths.append("Ideal BPM")
    elif scored["s_tempo"] < 40:
        warnings_list.append("BPM out of range")

    if scored["duration"] > 240:
        warnings_list.append(f"Long track ({scored['duration']:.0f}s)")

    if scored["s_loud"] < 40:
        warnings_list.append("Loudness off-target")

    return strengths, warnings_list


# ---------------------------------------------------------------------------
# Ordering algorithm
# ---------------------------------------------------------------------------

def optimal_order(scored_list: list[dict]) -> list[dict]:
    """
    Arrange tracks following streaming playlist psychology.
    Returns ordered list with 'position_rationale' key added.
    """
    tracks = [dict(t) for t in scored_list]
    n = len(tracks)
    ordered = []
    remaining = list(range(n))

    def pick(indices, key_fn, reverse=True):
        best = sorted(indices, key=key_fn, reverse=reverse)[0]
        remaining.remove(best)
        return best

    def is_consecutive_clash(prev: dict, cand: dict) -> bool:
        bpm_clash = abs(prev["tempo"] - cand["tempo"]) <= 3
        centroid_clash = abs(prev["centroid_mean"] - cand["centroid_mean"]) < 200
        key_clash = prev["key"] == cand["key"]
        return bpm_clash and centroid_clash and key_clash

    def best_candidate(indices, key_fn, prev=None):
        candidates = sorted(indices, key=key_fn, reverse=True)
        for idx in candidates:
            if prev is None or not is_consecutive_clash(prev, tracks[idx]):
                return idx
        return candidates[0]  # fallback

    # Track 1: highest intro_impact + high energy
    idx = best_candidate(
        remaining,
        lambda i: tracks[i]["intro_impact_raw"] * 0.6 + tracks[i]["energy_level"] * 0.4,
    )
    remaining.remove(idx)
    tracks[idx]["position_rationale"] = "Opener: highest intro impact & energy to hook listener"
    ordered.append(tracks[idx])

    # Track 2: maintain/escalate energy
    idx = best_candidate(
        remaining,
        lambda i: tracks[i]["energy_level"],
        prev=ordered[-1],
    )
    remaining.remove(idx)
    tracks[idx]["position_rationale"] = "Momentum: sustains or escalates opening energy"
    ordered.append(tracks[idx])

    # Tracks 3–min(5, n-3): peak energy zone
    peak_count = min(3, max(1, n - 4))
    for k in range(peak_count):
        if not remaining:
            break
        idx = best_candidate(
            remaining,
            lambda i: tracks[i]["energy_level"] * 0.5 + tracks[i]["total"] * 0.5,
            prev=ordered[-1],
        )
        remaining.remove(idx)
        tracks[idx]["position_rationale"] = f"Peak zone (position {len(ordered)+1}): high energy + strong score"
        ordered.append(tracks[idx])

    # Mid-album: gradual energy dip (lowest energy tracks)
    mid_count = max(1, n - len(ordered) - 2)
    low_energy = sorted(remaining, key=lambda i: tracks[i]["energy_level"])
    for i in range(min(mid_count, len(low_energy))):
        idx = low_energy[i]
        if idx in remaining:
            remaining.remove(idx)
            tracks[idx]["position_rationale"] = "Emotional valley: softer/slower track for contrast"
            ordered.append(tracks[idx])

    # Pre-closer: energy rebuild — pick highest energy from remaining
    if len(remaining) > 1:
        idx = best_candidate(
            remaining,
            lambda i: tracks[i]["energy_level"],
            prev=ordered[-1],
        )
        remaining.remove(idx)
        tracks[idx]["position_rationale"] = "Pre-closer: energy rebuild before finale"
        ordered.append(tracks[idx])

    # Closer: anthemic (high energy+score) OR emotional resolution (low energy)
    if remaining:
        # Decide by total score gap: if high-scoring remain, go anthemic
        top_remain = sorted(remaining, key=lambda i: tracks[i]["total"], reverse=True)
        idx = top_remain[0]
        remaining.remove(idx)
        if tracks[idx]["energy_level"] > np.median([t["energy_level"] for t in scored_list]):
            rationale = "Closer: anthemic peak — sends listener off on a high"
        else:
            rationale = "Closer: emotional resolution — slow, reflective finale"
        tracks[idx]["position_rationale"] = rationale
        ordered.append(tracks[idx])

    # Append any leftover (edge cases)
    for idx in remaining:
        tracks[idx]["position_rationale"] = "Filler position"
        ordered.append(tracks[idx])

    return ordered


# ---------------------------------------------------------------------------
# ASCII energy arc
# ---------------------------------------------------------------------------

def ascii_energy_arc(ordered: list[dict], width: int = 60) -> str:
    levels = [t["energy_level"] for t in ordered]
    lo, hi = min(levels), max(levels)
    rows = 10
    lines = []
    normalized = [
        int((v - lo) / (hi - lo + 1e-9) * (rows - 1)) for v in levels
    ]
    for row in range(rows - 1, -1, -1):
        line = ""
        for val in normalized:
            line += "█" if val >= row else " "
            line += " "
        label = f" {(row / (rows-1) * 100):3.0f}%"
        lines.append(line + label)
    # x-axis
    lines.append("─" * (len(normalized) * 2))
    track_nums = "".join(
        str(i + 1).ljust(2) for i in range(len(ordered))
    )
    lines.append(track_nums)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def collect_mp3s(directory: str) -> list[str]:
    p = Path(directory)
    files = sorted(p.glob("*.mp3")) + sorted(p.glob("*.MP3"))
    return [str(f) for f in files]


def fmt_duration(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m}:{sec:02d}"


@click.group()
def cli():
    """spotify_analyzer — Spotify track ranking & album ordering tool."""


@cli.command()
@click.argument("directory", default=".", type=click.Path(exists=True))
@click.option("--output", "-o", default="rank_result.csv", show_default=True,
              help="CSV output filename")
def rank(directory, output):
    """Rank MP3 files by Spotify performance potential."""
    mp3s = collect_mp3s(directory)
    if len(mp3s) < 2:
        console.print("[red]Need at least 2 MP3 files in the directory.[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold cyan]spotify_analyzer[/bold cyan] › rank mode")
    console.print(f"Found [yellow]{len(mp3s)}[/yellow] MP3 files in [dim]{directory}[/dim]\n")

    scored_list = []
    with console.status("[bold green]Analyzing tracks…") as status:
        for path in mp3s:
            status.update(f"[bold green]Analyzing:[/bold green] {Path(path).name}")
            feat = extract_features(path)
            scored = compute_scores(feat)
            scored_list.append(scored)

    scored_list.sort(key=lambda x: x["total"], reverse=True)

    table = Table(
        title="Spotify Track Ranking",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("Rank", justify="center", style="bold yellow", width=5)
    table.add_column("Filename", style="cyan", max_width=30)
    table.add_column("Score", justify="right", style="bold green", width=7)
    table.add_column("BPM", justify="right", width=6)
    table.add_column("Duration", justify="right", width=9)
    table.add_column("Centroid", justify="right", width=9)
    table.add_column("Strengths", style="green", max_width=28)
    table.add_column("Warnings", style="red", max_width=28)

    csv_rows = []
    for rank_pos, s in enumerate(scored_list, 1):
        strengths, warns = strengths_warnings(s)
        table.add_row(
            str(rank_pos),
            s["filename"],
            f"{s['total']:.1f}",
            f"{s['tempo']:.0f}",
            fmt_duration(s["duration"]),
            f"{s['centroid_mean']:.0f} Hz",
            ", ".join(strengths) or "—",
            ", ".join(warns) or "—",
        )
        csv_rows.append({
            "rank": rank_pos,
            "filename": s["filename"],
            "score": round(s["total"], 2),
            "bpm": round(s["tempo"], 1),
            "duration_s": round(s["duration"], 1),
            "centroid_hz": round(s["centroid_mean"], 1),
            "intro_score": round(s["s_intro"], 1),
            "brightness_score": round(s["s_bright"], 1),
            "energy_build_score": round(s["s_build"], 1),
            "dynamic_range_score": round(s["s_dyn"], 1),
            "tempo_score": round(s["s_tempo"], 1),
            "duration_score": round(s["s_dur"], 1),
            "loudness_score": round(s["s_loud"], 1),
            "strengths": "; ".join(strengths),
            "warnings": "; ".join(warns),
        })

    console.print(table)

    out_path = Path(directory) / output
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    console.print(f"\n[dim]Results exported → [bold]{out_path}[/bold][/dim]\n")


@cli.command()
@click.argument("directory", default=".", type=click.Path(exists=True))
@click.option("--output", "-o", default="album_order.csv", show_default=True,
              help="CSV output filename")
def order(directory, output):
    """Determine optimal Spotify album track order (10–16 MP3 files)."""
    mp3s = collect_mp3s(directory)
    if len(mp3s) < 2:
        console.print("[red]Need at least 2 MP3 files in the directory.[/red]")
        raise SystemExit(1)

    console.print(f"\n[bold cyan]spotify_analyzer[/bold cyan] › order mode")
    console.print(f"Found [yellow]{len(mp3s)}[/yellow] MP3 files in [dim]{directory}[/dim]\n")

    scored_list = []
    with console.status("[bold green]Analyzing tracks…") as status:
        for path in mp3s:
            status.update(f"[bold green]Analyzing:[/bold green] {Path(path).name}")
            feat = extract_features(path)
            scored = compute_scores(feat)
            scored_list.append(scored)

    ordered = optimal_order(scored_list)

    # Rich table
    table = Table(
        title="Optimal Album Track Order",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("#", justify="center", style="bold yellow", width=4)
    table.add_column("Filename", style="cyan", max_width=30)
    table.add_column("BPM", justify="right", width=6)
    table.add_column("Key", justify="center", width=5)
    table.add_column("Energy", justify="right", width=8)
    table.add_column("Mood", justify="right", width=7)
    table.add_column("Rationale", max_width=38)

    csv_rows = []
    for pos, t in enumerate(ordered, 1):
        energy_bar = "█" * int(t["energy_level"] * 100 / 0.3 * 8)
        table.add_row(
            str(pos),
            t["filename"],
            f"{t['tempo']:.0f}",
            t["key"],
            f"{t['energy_level']:.4f}",
            f"{t['valence']:.2f}",
            t.get("position_rationale", ""),
        )
        csv_rows.append({
            "position": pos,
            "filename": t["filename"],
            "bpm": round(t["tempo"], 1),
            "key": t["key"],
            "energy_level": round(t["energy_level"], 5),
            "valence": round(t["valence"], 3),
            "duration_s": round(t["duration"], 1),
            "rationale": t.get("position_rationale", ""),
        })

    console.print(table)

    # ASCII energy arc
    console.print("\n[bold]Energy Arc[/bold] (track order → energy level)\n")
    arc = ascii_energy_arc(ordered)
    console.print(arc)

    out_path = Path(directory) / output
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)

    console.print(f"\n[dim]Results exported → [bold]{out_path}[/bold][/dim]\n")


if __name__ == "__main__":
    cli()
