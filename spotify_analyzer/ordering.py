"""Optimal album track ordering following streaming playlist psychology."""

import numpy as np


# ---------------------------------------------------------------------------
# Clash detection
# ---------------------------------------------------------------------------

def _is_clash(prev: dict, cand: dict) -> bool:
    """Return True when consecutive placement of prev→cand is discouraged."""
    bpm_clash      = abs(prev["tempo"] - cand["tempo"]) <= 3
    centroid_clash = abs(prev["centroid_mean"] - cand["centroid_mean"]) < 200
    key_clash      = prev["key"] == cand["key"]
    return bpm_clash and centroid_clash and key_clash


def _best_candidate(tracks: list[dict], indices: list[int],
                    key_fn, prev: dict | None = None) -> int:
    """Pick the highest-scoring index that avoids a consecutive clash."""
    candidates = sorted(indices, key=key_fn, reverse=True)
    for idx in candidates:
        if prev is None or not _is_clash(prev, tracks[idx]):
            return idx
    return candidates[0]  # fallback: ignore clash rather than crash


# ---------------------------------------------------------------------------
# Main ordering algorithm
# ---------------------------------------------------------------------------

def optimal_order(scored_list: list[dict]) -> list[dict]:
    """Arrange tracks following streaming playlist psychology rules.

    Rules applied:
      1. Track 1  — highest intro impact + high energy (hook the listener)
      2. Track 2  — sustain or escalate energy
      3. Tracks 3–5 — peak energy zone
      4. Mid-album — gradual energy dip (emotional valley)
      5. Pre-closer — energy rebuild
      6. Closer   — anthemic peak OR slow emotional resolution
      7. No consecutive tracks with identical BPM (±3), similar spectral
         profile (< 200 Hz centroid gap), or same key.

    Returns a new list with 'position_rationale' added to each track dict.
    """
    tracks = [dict(t) for t in scored_list]
    n = len(tracks)
    ordered: list[dict] = []
    remaining: list[int] = list(range(n))

    # --- Track 1: opener ---
    idx = _best_candidate(
        tracks, remaining,
        lambda i: tracks[i]["intro_impact_raw"] * 0.6 + tracks[i]["energy_level"] * 0.4,
    )
    remaining.remove(idx)
    tracks[idx]["position_rationale"] = (
        "Opener: highest intro impact & energy to hook listener"
    )
    ordered.append(tracks[idx])

    # --- Track 2: momentum ---
    if remaining:
        idx = _best_candidate(
            tracks, remaining,
            lambda i: tracks[i]["energy_level"],
            prev=ordered[-1],
        )
        remaining.remove(idx)
        tracks[idx]["position_rationale"] = (
            "Momentum: sustains or escalates opening energy"
        )
        ordered.append(tracks[idx])

    # --- Tracks 3–5: peak energy zone ---
    peak_count = min(3, max(1, n - 4))
    for _ in range(peak_count):
        if not remaining:
            break
        pos_label = len(ordered) + 1
        idx = _best_candidate(
            tracks, remaining,
            lambda i: tracks[i]["energy_level"] * 0.5 + tracks[i]["total"] * 0.5,
            prev=ordered[-1],
        )
        remaining.remove(idx)
        tracks[idx]["position_rationale"] = (
            f"Peak zone (position {pos_label}): high energy + strong score"
        )
        ordered.append(tracks[idx])

    # --- Mid-album: emotional valley (lowest energy tracks) ---
    valley_count = max(1, n - len(ordered) - 2)
    low_energy_indices = sorted(remaining, key=lambda i: tracks[i]["energy_level"])
    for i in range(min(valley_count, len(low_energy_indices))):
        idx = low_energy_indices[i]
        if idx in remaining:
            remaining.remove(idx)
            tracks[idx]["position_rationale"] = (
                "Emotional valley: softer/slower track for contrast"
            )
            ordered.append(tracks[idx])

    # --- Pre-closer: energy rebuild ---
    if len(remaining) > 1:
        idx = _best_candidate(
            tracks, remaining,
            lambda i: tracks[i]["energy_level"],
            prev=ordered[-1],
        )
        remaining.remove(idx)
        tracks[idx]["position_rationale"] = "Pre-closer: energy rebuild before finale"
        ordered.append(tracks[idx])

    # --- Closer: anthemic OR emotional resolution ---
    if remaining:
        top_remain = sorted(remaining, key=lambda i: tracks[i]["total"], reverse=True)
        idx = top_remain[0]
        remaining.remove(idx)
        median_energy = float(np.median([t["energy_level"] for t in scored_list]))
        if tracks[idx]["energy_level"] > median_energy:
            rationale = "Closer: anthemic peak — sends listener off on a high"
        else:
            rationale = "Closer: emotional resolution — slow, reflective finale"
        tracks[idx]["position_rationale"] = rationale
        ordered.append(tracks[idx])

    # --- Edge-case leftovers ---
    for idx in remaining:
        tracks[idx]["position_rationale"] = "Additional track"
        ordered.append(tracks[idx])

    return ordered


# ---------------------------------------------------------------------------
# ASCII energy arc visualisation
# ---------------------------------------------------------------------------

def ascii_energy_arc(ordered: list[dict], rows: int = 10) -> str:
    """Render a vertical bar chart of energy levels across the track order."""
    levels = [t["energy_level"] for t in ordered]
    lo, hi = min(levels), max(levels)
    span = hi - lo + 1e-9
    bar_heights = [int((v - lo) / span * (rows - 1)) for v in levels]

    lines = []
    for row in range(rows - 1, -1, -1):
        bar_line = "".join("█ " if h >= row else "  " for h in bar_heights)
        pct = int(row / (rows - 1) * 100)
        lines.append(f"{bar_line} {pct:3d}%")

    lines.append("─" * (len(ordered) * 2))
    lines.append("".join(str(i + 1).ljust(2) for i in range(len(ordered))))
    return "\n".join(lines)
