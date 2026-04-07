"""End-to-end CLI tests using Click's test runner."""

import csv
from pathlib import Path

import pytest
from click.testing import CliRunner

from spotify_analyzer.cli import cli


@pytest.fixture
def runner():
    return CliRunner()


# ---------------------------------------------------------------------------
# --help / --version
# ---------------------------------------------------------------------------

class TestHelp:
    def test_main_help(self, runner):
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "rank" in result.output
        assert "order" in result.output

    def test_rank_help(self, runner):
        result = runner.invoke(cli, ["rank", "--help"])
        assert result.exit_code == 0
        assert "DIRECTORY" in result.output

    def test_order_help(self, runner):
        result = runner.invoke(cli, ["order", "--help"])
        assert result.exit_code == 0
        assert "DIRECTORY" in result.output


# ---------------------------------------------------------------------------
# rank command
# ---------------------------------------------------------------------------

class TestRankCommand:
    def test_rank_produces_table_and_csv(self, runner, tmp_audio_dir):
        result = runner.invoke(cli, ["rank", tmp_audio_dir])
        assert result.exit_code == 0, result.output
        assert "Spotify Track Ranking" in result.output
        csv_path = Path(tmp_audio_dir) / "rank_result.csv"
        assert csv_path.exists()

    def test_rank_csv_has_expected_columns(self, runner, tmp_audio_dir):
        runner.invoke(cli, ["rank", tmp_audio_dir])
        csv_path = Path(tmp_audio_dir) / "rank_result.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
        for col in ("rank", "filename", "score", "bpm", "duration_s"):
            assert col in cols, col

    def test_rank_custom_output_name(self, runner, tmp_audio_dir):
        result = runner.invoke(cli, ["rank", tmp_audio_dir, "-o", "custom_rank.csv"])
        assert result.exit_code == 0, result.output
        assert (Path(tmp_audio_dir) / "custom_rank.csv").exists()

    def test_rank_csv_row_count_matches_files(self, runner, tmp_audio_dir):
        runner.invoke(cli, ["rank", tmp_audio_dir])
        mp3_count = len(list(Path(tmp_audio_dir).glob("*.mp3")))
        csv_path = Path(tmp_audio_dir) / "rank_result.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == mp3_count

    def test_rank_scores_are_ordered_descending(self, runner, tmp_audio_dir):
        runner.invoke(cli, ["rank", tmp_audio_dir])
        csv_path = Path(tmp_audio_dir) / "rank_result.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            scores = [float(row["score"]) for row in csv.DictReader(f)]
        assert scores == sorted(scores, reverse=True)

    def test_rank_fails_with_empty_dir(self, runner, tmp_path):
        result = runner.invoke(cli, ["rank", str(tmp_path)])
        assert result.exit_code != 0

    def test_rank_fails_with_one_file(self, runner, tmp_audio_dir, tmp_path):
        import shutil
        mp3 = next(Path(tmp_audio_dir).glob("*.mp3"))
        shutil.copy(str(mp3), str(tmp_path / mp3.name))
        result = runner.invoke(cli, ["rank", str(tmp_path)])
        assert result.exit_code != 0


# ---------------------------------------------------------------------------
# order command
# ---------------------------------------------------------------------------

class TestOrderCommand:
    def test_order_produces_table_and_csv(self, runner, tmp_audio_dir):
        result = runner.invoke(cli, ["order", tmp_audio_dir])
        assert result.exit_code == 0, result.output
        assert "Optimal Album Track Order" in result.output
        csv_path = Path(tmp_audio_dir) / "album_order.csv"
        assert csv_path.exists()

    def test_order_csv_has_expected_columns(self, runner, tmp_audio_dir):
        runner.invoke(cli, ["order", tmp_audio_dir])
        csv_path = Path(tmp_audio_dir) / "album_order.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            cols = csv.DictReader(f).fieldnames or []
        for col in ("position", "filename", "bpm", "key", "energy_level", "rationale"):
            assert col in cols, col

    def test_order_csv_positions_are_sequential(self, runner, tmp_audio_dir):
        runner.invoke(cli, ["order", tmp_audio_dir])
        csv_path = Path(tmp_audio_dir) / "album_order.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            positions = [int(row["position"]) for row in csv.DictReader(f)]
        assert positions == list(range(1, len(positions) + 1))

    def test_order_no_duplicate_positions(self, runner, tmp_audio_dir):
        runner.invoke(cli, ["order", tmp_audio_dir])
        csv_path = Path(tmp_audio_dir) / "album_order.csv"
        with open(csv_path, newline="", encoding="utf-8") as f:
            positions = [row["position"] for row in csv.DictReader(f)]
        assert len(positions) == len(set(positions))

    def test_order_ascii_arc_in_output(self, runner, tmp_audio_dir):
        result = runner.invoke(cli, ["order", tmp_audio_dir])
        assert result.exit_code == 0
        assert "Energy Arc" in result.output
        assert "█" in result.output

    def test_order_custom_output_name(self, runner, tmp_audio_dir):
        result = runner.invoke(cli, ["order", tmp_audio_dir, "-o", "my_album.csv"])
        assert result.exit_code == 0, result.output
        assert (Path(tmp_audio_dir) / "my_album.csv").exists()

    def test_order_fails_with_empty_dir(self, runner, tmp_path):
        result = runner.invoke(cli, ["order", str(tmp_path)])
        assert result.exit_code != 0
