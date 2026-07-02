from pathlib import Path

from PIL import Image

from roughcut.text_render import render_overlay_png, _strip_dashes, TARGET_W, TARGET_H
from roughcut.schemas import TextOverlay


def test_strip_em_dash():
    assert "—" not in _strip_dashes("before — after")
    assert _strip_dashes("a — b") == "a, b"


def test_strip_en_dash():
    assert "–" not in _strip_dashes("a – b")


def test_no_double_commas():
    out = _strip_dashes("x —, y")
    assert ",," not in out


def test_render_produces_full_width_png(tmp_path: Path):
    ov = TextOverlay(type="hook", text="hello world", start_s=0, duration_s=2)
    path, y = render_overlay_png(ov, tmp_path / "o.png")
    img = Image.open(path)
    assert img.width == TARGET_W
    assert img.mode == "RGBA"
    assert 0 <= y <= TARGET_H - img.height


def test_long_text_wraps(tmp_path: Path):
    long_text = "this is a very long hook line that absolutely must wrap to multiple lines to stay readable"
    ov = TextOverlay(type="hook", text=long_text, start_s=0, duration_s=2)
    path, _ = render_overlay_png(ov, tmp_path / "long.png")
    img = Image.open(path)
    # Multi-line render → canvas meaningfully taller than single line
    assert img.height > 150


def test_watermark_anchors_near_bottom(tmp_path: Path):
    ov = TextOverlay(type="lower_third", text="made with roughcut", start_s=0, duration_s=20, style="watermark")
    _, y = render_overlay_png(ov, tmp_path / "w.png")
    assert y > TARGET_H * 0.85


def test_bold_center_is_upper_third(tmp_path: Path):
    ov = TextOverlay(type="hook", text="hook text", start_s=0, duration_s=2, style="bold_center")
    _, y = render_overlay_png(ov, tmp_path / "h.png")
    assert y < TARGET_H * 0.35
