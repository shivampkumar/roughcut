from pathlib import Path

from roughcut.script import load_script, align_script, _similarity
from roughcut.schemas import ClipAnalysis, SpeechSegment


def test_load_script_strips_stage_directions(tmp_path: Path):
    p = tmp_path / "s.txt"
    p.write_text(
        "[OPENING]\nTonight we found something incredible.\n\n"
        "[CUE: reveal]\nKesha walked in and grabbed the mic.\n"
    )
    sc = load_script(p)
    assert len(sc.beats) == 2
    assert "[" not in sc.beats[0].text
    assert sc.beats[0].text.startswith("Tonight")


def test_load_script_splits_sentences_when_no_blank_lines(tmp_path: Path):
    p = tmp_path / "s.txt"
    p.write_text("First sentence. Second sentence! Third?")
    sc = load_script(p)
    assert len(sc.beats) == 3


def test_similarity_basic():
    assert _similarity("hello world", "hello world") == 1.0
    assert _similarity("hello world", "completely different") < 0.5


def _mk_analysis(path: str, shot: str) -> ClipAnalysis:
    return ClipAnalysis(
        asset_path=Path(path),
        action="x", emotion="x", setting="x",
        composition_score=5, aesthetic_score=5, energy=5,
        narrative_role="setup", shot_type=shot,
    )


def test_align_script_matches_a_roll(tmp_path: Path):
    p = tmp_path / "s.txt"
    p.write_text("We drove eight hundred miles to get here.")
    sc = load_script(p)

    a_roll = Path("/fake/a_roll.mp4")
    speech = {
        a_roll: [
            SpeechSegment(start_s=2.0, end_s=5.5, text="we drove eight hundred miles to get here"),
            SpeechSegment(start_s=6.0, end_s=8.0, text="and it was worth it"),
        ]
    }
    analyses = [_mk_analysis(str(a_roll), "a_roll")]
    aligned = align_script(sc, speech, analyses)
    beat = aligned.beats[0]
    assert beat.a_roll_asset_path == a_roll
    assert beat.match_score > 0.8
    assert beat.a_roll_in_s == 2.0


def test_align_script_unmatched_beat_stays_none(tmp_path: Path):
    p = tmp_path / "s.txt"
    p.write_text("Totally unrelated content about quantum physics.")
    sc = load_script(p)
    speech = {
        Path("/fake/clip.mp4"): [
            SpeechSegment(start_s=0.0, end_s=3.0, text="happy birthday to you"),
        ]
    }
    aligned = align_script(sc, speech, [])
    assert aligned.beats[0].a_roll_asset_path is None
