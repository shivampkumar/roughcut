from pathlib import Path

from roughcut.crop import plan_crop_for_clip, TARGET_AR
from roughcut.schemas import EDLClip, MediaAsset


def _photo(w: int, h: int) -> MediaAsset:
    return MediaAsset(path=Path("/fake.jpg"), type="photo", width=w, height=h)


def _clip(hint: str = "center") -> EDLClip:
    return EDLClip(asset_path=Path("/fake.jpg"), in_s=0, out_s=2, crop_hint=hint)


def test_wide_source_crops_horizontally():
    plan = plan_crop_for_clip(_clip(), _photo(1920, 1080))
    assert plan.crop_h == 1080
    assert plan.crop_w == round(1080 * TARGET_AR)
    assert 0 <= plan.crop_x <= 1920 - plan.crop_w


def test_tall_source_crops_vertically():
    # 9:21 source taller than 9:16 → crop vertically
    plan = plan_crop_for_clip(_clip(), _photo(720, 1680))
    assert plan.crop_w == 720
    assert plan.crop_h == round(720 / TARGET_AR)
    assert 0 <= plan.crop_y <= 1680 - plan.crop_h


def test_face_left_hint_anchors_left():
    left = plan_crop_for_clip(_clip("face_left"), _photo(1920, 1080))
    right = plan_crop_for_clip(_clip("face_right"), _photo(1920, 1080))
    center = plan_crop_for_clip(_clip("center"), _photo(1920, 1080))
    assert left.crop_x < center.crop_x < right.crop_x


def test_crop_clamped_to_frame():
    plan = plan_crop_for_clip(_clip("face_right"), _photo(700, 1200))
    assert plan.crop_x + plan.crop_w <= 700
    assert plan.crop_y + plan.crop_h <= 1200
    assert plan.crop_x >= 0 and plan.crop_y >= 0


def test_exact_916_source_full_frame():
    plan = plan_crop_for_clip(_clip(), _photo(1080, 1920))
    assert plan.crop_w == 1080
    assert plan.crop_h == 1920
    assert plan.crop_x == 0 and plan.crop_y == 0


def test_face_detect_never_raises(monkeypatch):
    # Even with broken detectors (e.g. OpenCV 5.x missing cascade file),
    # face detection must return None, not crash the render.
    import numpy as np
    from roughcut import crop as crop_mod

    def boom(frame):
        raise RuntimeError("detector exploded")

    monkeypatch.setattr(crop_mod, "_detect_face_box_mediapipe", boom)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    assert crop_mod._detect_face_box(frame) is None
