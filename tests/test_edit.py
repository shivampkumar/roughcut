from pathlib import Path

from roughcut.edit import (
    AddOverlay,
    AddSFX,
    RemoveClip,
    RemoveOverlay,
    TrimClip,
    UpdateClipAudio,
    apply_edits,
    apply_one,
)
from roughcut.schemas import EDL, EDLClip, TextOverlay


def _mk_edl() -> EDL:
    return EDL(
        event_type="party",
        arc=[
            EDLClip(asset_path=Path("/a.mp4"), in_s=0, out_s=3, role="hook"),
            EDLClip(asset_path=Path("/b.mp4"), in_s=5, out_s=8, role="climax"),
        ],
        music_mood="hype",
        music_genre="pop",
        overlays=[
            TextOverlay(type="hook", text="hello world", start_s=0, duration_s=2),
        ],
    )


def test_add_overlay():
    edl = apply_one(_mk_edl(), AddOverlay(text="WAIT", start_s=4.0))
    assert len(edl.overlays) == 2
    assert edl.overlays[1].text == "WAIT"


def test_remove_overlay_by_match():
    edl = apply_one(_mk_edl(), RemoveOverlay(match="hello"))
    assert len(edl.overlays) == 0


def test_trim_clip_delta():
    edl = apply_one(_mk_edl(), TrimClip(index=0, delta_out_s=-1.0))
    assert edl.arc[0].out_s == 2.0


def test_trim_clip_floor():
    # Trimming below 0.3s clamps
    edl = apply_one(_mk_edl(), TrimClip(index=0, new_out_s=0.05))
    assert edl.arc[0].duration_s >= 0.3


def test_remove_clip():
    edl = apply_one(_mk_edl(), RemoveClip(index=0))
    assert len(edl.arc) == 1
    assert edl.arc[0].asset_path == Path("/b.mp4")


def test_update_clip_audio():
    edl = apply_one(_mk_edl(), UpdateClipAudio(index=1, audio_decision="duck", gain_db=-8))
    assert edl.arc[1].audio_decision == "duck"
    assert edl.arc[1].original_audio_gain_db == -8


def test_apply_edits_chains():
    ops = [
        AddSFX(sfx="boom", at_s=2.0),
        RemoveClip(index=1),
        AddOverlay(text="END", start_s=2.5),
    ]
    edl = apply_edits(_mk_edl(), ops)
    assert len(edl.sfx) == 1
    assert len(edl.arc) == 1
    assert any(o.text == "END" for o in edl.overlays)


def test_original_edl_not_mutated():
    original = _mk_edl()
    apply_one(original, RemoveClip(index=0))
    assert len(original.arc) == 2
