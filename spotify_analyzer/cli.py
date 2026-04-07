"""Click CLI entry-point for spotify_analyzer."""

import csv
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table
from rich import box

from .features import extract_features
from .scoring import compute_scores, strengths_warnings
from .ordering import optimal_order, ascii_energy_arc

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _collect_audio(directory: str) -> list[str]:
    """Return sorted list of .mp3 / .MP3 paths in *directory*."""
    p = Path(directory)
    files = sorted(p.glob("*.mp3")) + sorted(p.glob("*.MP3"))
    return [str(f) for f in files]


def _fmt_duration(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _analyze_all(paths: list[str]) -> list[dict]:
    """Extract features + compute scores for every file, with progress."""
    results = []
    with console.status("[bold green]Analyzing tracks…") as status:
        for path in paths:
            status.update(f"[bold green]Analyzing:[/bold green] {Path(path).name}")
            feat = extract_features(path)
            scored = compute_scores(feat)
            results.append(scored)
    return results


def _write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="spotify-analyzer")
def cli():
    """spotify_analyzer — Spotify track ranking & album ordering tool.

    \b
    Modes:
      rank   Score 6–8 MP3 files for Spotify performance potential.
      order  Find optimal playback order for a 10–16 track album.

    \b
    Examples:
      spotify_analyzer rank ./tracks/
      spotify_analyzer rank ./tracks/ -o my_scores.csv
      spotify_analyzer order ./album/
      spotify_analyzer order ./album/ -o tracklist.csv
    """


# ---------------------------------------------------------------------------
# rank command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("directory", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default="rank_result.csv", show_default=True,
              help="CSV output filename (written inside DIRECTORY).")
def rank(directory: str, output: str) -> None:
    """Rank MP3 files in DIRECTORY by Spotify performance potential.

    \b
    Scoring weights:
      Intro impact      25%   (first 5-second RMS vs. track average)
      Spectral brightness 20% (target centroid: 3 300–4 100 Hz)
      Energy build      15%   (2nd-half vs 1st-half RMS delta)
      Tempo fit         15%   (ideal: 95–135 BPM for K-pop/indie-pop)
      Dynamic range     10%   (max − mean RMS)
      Duration fit      10%   (penalise tracks > 4:00)
      Loudness fit       5%   (RMS proxy for −14 LUFS target)
    """
    mp3s = _collect_audio(directory)
    if len(mp3s) < 2:
        console.print("[red]Error:[/red] Need at least 2 MP3 files in the directory.")
        raise SystemExit(1)

    console.print(f"\n[bold cyan]spotify_analyzer[/bold cyan] › [bold]rank[/bold] mode")
    console.print(
        f"Analyzing [yellow]{len(mp3s)}[/yellow] MP3 files in "
        f"[dim]{directory}[/dim]\n"
    )

    scored_list = _analyze_all(mp3s)
    scored_list.sort(key=lambda x: x["total"], reverse=True)

    # ── Rich table ──────────────────────────────────────────────────────────
    table = Table(
        title="Spotify Track Ranking",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("Rank",     justify="center",  style="bold yellow", width=5)
    table.add_column("Filename", style="cyan",       max_width=32)
    table.add_column("Score",    justify="right",    style="bold green",  width=7)
    table.add_column("BPM",      justify="right",    width=6)
    table.add_column("Duration", justify="right",    width=9)
    table.add_column("Centroid", justify="right",    width=10)
    table.add_column("Strengths", style="green",     max_width=28)
    table.add_column("Warnings",  style="red",       max_width=28)

    csv_rows = []
    for rank_pos, s in enumerate(scored_list, 1):
        strengths, warns = strengths_warnings(s)
        table.add_row(
            str(rank_pos),
            s["filename"],
            f"{s['total']:.1f}",
            f"{s['tempo']:.0f}",
            _fmt_duration(s["duration"]),
            f"{s['centroid_mean']:.0f} Hz",
            ", ".join(strengths) or "—",
            ", ".join(warns) or "—",
        )
        csv_rows.append({
            "rank":                rank_pos,
            "filename":            s["filename"],
            "score":               round(s["total"], 2),
            "bpm":                 round(s["tempo"], 1),
            "duration_s":          round(s["duration"], 1),
            "centroid_hz":         round(s["centroid_mean"], 1),
            "intro_score":         round(s["s_intro"], 1),
            "brightness_score":    round(s["s_bright"], 1),
            "energy_build_score":  round(s["s_build"], 1),
            "dynamic_range_score": round(s["s_dyn"], 1),
            "tempo_score":         round(s["s_tempo"], 1),
            "duration_score":      round(s["s_dur"], 1),
            "loudness_score":      round(s["s_loud"], 1),
            "strengths":           "; ".join(strengths),
            "warnings":            "; ".join(warns),
        })

    console.print(table)

    out_path = Path(directory) / output
    _write_csv(csv_rows, out_path)
    console.print(f"\n[dim]Results exported → [bold]{out_path}[/bold][/dim]\n")


# ---------------------------------------------------------------------------
# order command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("directory", default=".", type=click.Path(exists=True, file_okay=False))
@click.option("--output", "-o", default="album_order.csv", show_default=True,
              help="CSV output filename (written inside DIRECTORY).")
def order(directory: str, output: str) -> None:
    """Find optimal Spotify album track order for MP3s in DIRECTORY.

    \b
    Ordering rules:
      Track 1    — highest intro impact + energy (hook the listener)
      Track 2    — sustain or escalate energy
      Tracks 3–5 — peak energy zone
      Mid-album  — emotional valley (slower/softer tracks)
      Pre-closer — energy rebuild
      Closer     — anthemic peak OR slow emotional resolution
      Constraint — no consecutive tracks with matching BPM (±3 BPM),
                   similar spectral profile, or same musical key.
    """
    mp3s = _collect_audio(directory)
    if len(mp3s) < 2:
        console.print("[red]Error:[/red] Need at least 2 MP3 files in the directory.")
        raise SystemExit(1)

    console.print(f"\n[bold cyan]spotify_analyzer[/bold cyan] › [bold]order[/bold] mode")
    console.print(
        f"Analyzing [yellow]{len(mp3s)}[/yellow] MP3 files in "
        f"[dim]{directory}[/dim]\n"
    )

    scored_list = _analyze_all(mp3s)
    ordered = optimal_order(scored_list)

    # ── Rich table ──────────────────────────────────────────────────────────
    table = Table(
        title="Optimal Album Track Order",
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold magenta",
    )
    table.add_column("#",         justify="center", style="bold yellow", width=4)
    table.add_column("Filename",  style="cyan",     max_width=32)
    table.add_column("BPM",       justify="right",  width=6)
    table.add_column("Key",       justify="center", width=5)
    table.add_column("Energy",    justify="right",  width=8)
    table.add_column("Mood",      justify="right",  width=7)
    table.add_column("Rationale", max_width=40)

    csv_rows = []
    for pos, t in enumerate(ordered, 1):
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
            "position":     pos,
            "filename":     t["filename"],
            "bpm":          round(t["tempo"], 1),
            "key":          t["key"],
            "energy_level": round(t["energy_level"], 5),
            "valence":      round(t["valence"], 3),
            "duration_s":   round(t["duration"], 1),
            "rationale":    t.get("position_rationale", ""),
        })

    console.print(table)

    # ── ASCII energy arc ────────────────────────────────────────────────────
    console.print("\n[bold]Energy Arc[/bold]  (track position → relative energy level)\n")
    console.print(ascii_energy_arc(ordered))

    out_path = Path(directory) / output
    _write_csv(csv_rows, out_path)
    console.print(f"\n[dim]Results exported → [bold]{out_path}[/bold][/dim]\n")
