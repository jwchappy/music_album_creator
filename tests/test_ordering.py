"""Unit tests for spotify_analyzer.ordering."""

import pytest
from spotify_analyzer.ordering import optimal_order, ascii_energy_arc


# ---------------------------------------------------------------------------
# optimal_order
# ---------------------------------------------------------------------------

class TestOptimalOrder:
    def test_returns_all_tracks(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        assert len(ordered) == len(scored_list_8)

    def test_no_duplicate_tracks(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        filenames = [t["filename"] for t in ordered]
        assert len(filenames) == len(set(filenames))

    def test_every_track_has_rationale(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        for t in ordered:
            assert "position_rationale" in t
            assert len(t["position_rationale"]) > 0

    def test_opener_rationale_mentions_opener(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        assert "Opener" in ordered[0]["position_rationale"]

    def test_second_track_rationale_mentions_momentum(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        assert "Momentum" in ordered[1]["position_rationale"]

    def test_original_list_not_mutated(self, scored_list_8):
        originals = [t["filename"] for t in scored_list_8]
        optimal_order(scored_list_8)
        assert [t["filename"] for t in scored_list_8] == originals

    def test_works_with_two_tracks(self, sample_feat):
        from spotify_analyzer.scoring import compute_scores
        tracks = [
            compute_scores({**sample_feat, "filename": "a.mp3", "tempo": 100}),
            compute_scores({**sample_feat, "filename": "b.mp3", "tempo": 120,
                            "centroid_mean": 3800.0, "key": "D"}),
        ]
        ordered = optimal_order(tracks)
        assert len(ordered) == 2

    def test_works_with_sixteen_tracks(self, sample_feat):
        from spotify_analyzer.scoring import compute_scores
        keys = ["C", "C#", "D", "D#", "E", "F", "F#", "G",
                "G#", "A", "A#", "B", "C", "D", "E", "F"]
        tracks = [
            compute_scores({
                **sample_feat,
                "filename": f"t{i:02d}.mp3",
                "tempo": 80 + i * 5,
                "energy_level": 0.05 + i * 0.01,
                "rms_mean": 0.05 + i * 0.01,
                "centroid_mean": 1500 + i * 200,
                "key": keys[i],
            })
            for i in range(16)
        ]
        ordered = optimal_order(tracks)
        assert len(ordered) == 16

    def test_no_clash_between_consecutive_tracks(self, scored_list_8):
        """Verify the algorithm avoids BPM+centroid+key triple clashes where possible."""
        ordered = optimal_order(scored_list_8)
        clash_count = 0
        for prev, curr in zip(ordered, ordered[1:]):
            bpm_clash = abs(prev["tempo"] - curr["tempo"]) <= 3
            centroid_clash = abs(prev["centroid_mean"] - curr["centroid_mean"]) < 200
            key_clash = prev["key"] == curr["key"]
            if bpm_clash and centroid_clash and key_clash:
                clash_count += 1
        # With 8 distinct tracks the algorithm should produce zero full clashes
        assert clash_count == 0


# ---------------------------------------------------------------------------
# ascii_energy_arc
# ---------------------------------------------------------------------------

class TestAsciiEnergyArc:
    def test_returns_string(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        arc = ascii_energy_arc(ordered)
        assert isinstance(arc, str)

    def test_contains_block_character(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        arc = ascii_energy_arc(ordered)
        assert "█" in arc

    def test_has_correct_number_of_track_labels(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        arc = ascii_energy_arc(ordered)
        last_line = arc.splitlines()[-1]
        # Last line has one label per track
        labels = last_line.split()
        assert len(labels) == len(ordered)

    def test_single_track_no_crash(self, sample_feat):
        from spotify_analyzer.scoring import compute_scores
        single = [compute_scores(sample_feat)]
        arc = ascii_energy_arc(single)
        assert isinstance(arc, str)

    def test_rows_parameter(self, scored_list_8):
        ordered = optimal_order(scored_list_8)
        arc5 = ascii_energy_arc(ordered, rows=5)
        arc15 = ascii_energy_arc(ordered, rows=15)
        # More rows → more lines
        assert len(arc15.splitlines()) > len(arc5.splitlines())
