from pathlib import Path

from roughcut.music import TrackAnalysis, snap_edl_to_beats
from roughcut.schemas import EDL, EDLClip


def _track(beats: list[float]) -> TrackAnalysis:
    return TrackAnalysis(
        path=Path("/fake.wav"),
        duration_s=beats[-1] + 1 if beats else 10,
        bpm=120.0,
        beat_times=beats,
        downbeat_times=beats[::4],
        energy_curve=[0.5] * 10,
    )


def _edl(durations: list[float]) -> EDL:
    arc, t = [], 0.0
    for d in durations:
        arc.append(EDLClip(asset_path=Path("/c.mp4"), in_s=0, out_s=d))
        t += d
    return EDL(event_type="test", arc=arc, music_mood="x", music_genre="x")


def test_full_snap_lands_cuts_on_beats():
    beats = [i * 0.5 for i in range(40)]  # 120 BPM
    edl = snap_edl_to_beats(_edl([1.2, 0.8, 2.3]), _track(beats), snap_strength=1.0)
    t = 0.0
    for clip in edl.arc:
        t += clip.duration_s
        nearest = min(beats, key=lambda b: abs(b - t))
        assert abs(t - nearest) < 1e-6, f"cut at {t} not on a beat"


def test_zero_snap_changes_nothing():
    original = _edl([1.2, 0.8])
    snapped = snap_edl_to_beats(original, _track([i * 0.5 for i in range(20)]), snap_strength=0.0)
    for a, b in zip(original.arc, snapped.arc):
        assert abs(a.duration_s - b.duration_s) < 1e-6


def test_no_beats_is_noop():
    original = _edl([1.5])
    snapped = snap_edl_to_beats(original, _track([]), snap_strength=1.0)
    assert snapped.arc[0].duration_s == original.arc[0].duration_s


def test_min_duration_floor():
    # Clip so short that snapping would collapse it — floor at 0.3s
    edl = snap_edl_to_beats(_edl([0.05]), _track([0.0, 2.0]), snap_strength=1.0)
    assert edl.arc[0].duration_s >= 0.3


def test_analyze_track_on_real_wav(tmp_path):
    # Generate a click track and verify librosa finds the right tempo ballpark
    import numpy as np
    import soundfile as sf
    from roughcut.music import analyze_track

    sr = 22050
    dur = 10.0
    audio = np.zeros(int(sr * dur))
    for b in np.arange(0, dur, 0.5):  # 120 BPM clicks
        n0 = int(b * sr)
        click = np.sin(2 * np.pi * 1000 * np.arange(int(0.01 * sr)) / sr)
        audio[n0 : n0 + len(click)] += click
    p = tmp_path / "click.wav"
    sf.write(str(p), audio, sr)

    t = analyze_track(p)
    assert 100 < t.bpm < 140, f"expected ~120 BPM, got {t.bpm}"
    assert len(t.beat_times) > 10
