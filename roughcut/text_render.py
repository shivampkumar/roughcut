"""PIL-based text overlay rendering.

ffmpeg's drawtext filter needs libfreetype which isn't always built in.
This module renders each TextOverlay to a transparent PNG with proper Unicode,
stroke, and shadow. The PNGs are then composited into the video via ffmpeg's
overlay filter — universally available.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from roughcut.schemas import TextOverlay

TARGET_W = 1080
TARGET_H = 1920


_FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
    "/Library/Fonts/Arial Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _load_font(size: int) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size=size)
        except (OSError, IOError):
            continue
    return ImageFont.load_default()


def _style_params(style: str) -> dict:
    """Return rendering params per style preset."""
    if style == "bottom_sub":
        return {"fontsize": 60, "stroke": 5, "y_anchor": "bottom"}
    if style == "top_caption":
        return {"fontsize": 64, "stroke": 5, "y_anchor": "top"}
    if style == "watermark":
        return {"fontsize": 28, "stroke": 2, "y_anchor": "watermark"}
    if style == "safe_top":
        return {"fontsize": 84, "stroke": 6, "y_anchor": "safe_top"}
    # default: bold_center — upper third, not dead center (avoids covering faces)
    return {"fontsize": 88, "stroke": 6, "y_anchor": "upper_third"}


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int) -> list[str]:
    """Greedy word-wrap to keep text within max_width px."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for w in words:
        candidate = f"{current} {w}".strip()
        bbox = font.getbbox(candidate)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


def _strip_dashes(text: str) -> str:
    """Remove em/en dashes per project rule. Keep punctuation clean."""
    # Common shapes — try the spaced form first so we don't leave double spaces.
    for d in ("—", "–"):
        text = text.replace(f" {d} ", ", ")
        text = text.replace(f"{d} ", ", ")
        text = text.replace(f" {d}", ", ")
        text = text.replace(d, ", ")
    # Cleanup: collapse " ,"  and double commas
    text = text.replace(' ",', '",').replace(' .', '.')
    while ",," in text:
        text = text.replace(",,", ",")
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


def render_overlay_png(overlay: TextOverlay, out_path: Path) -> tuple[Path, int]:
    """Render the overlay text to a transparent PNG.

    Returns (path, y_offset_from_top_of_canvas). Canvas is 1080x1920.
    """
    p = _style_params(overlay.style)
    font = _load_font(p["fontsize"])
    max_text_width = int(TARGET_W * 0.88)
    clean_text = _strip_dashes(overlay.text)
    lines = _wrap_text(clean_text, font, max_text_width)

    # Compute total height
    line_metrics = [font.getbbox(line) for line in lines]
    line_heights = [(b[3] - b[1]) for b in line_metrics]
    line_spacing = int(p["fontsize"] * 0.18)
    total_h = sum(line_heights) + line_spacing * (len(lines) - 1)
    pad = p["stroke"] * 4
    canvas_h = total_h + pad * 2

    # Build transparent PNG sized 1080 x canvas_h
    img = Image.new("RGBA", (TARGET_W, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    y = pad
    for line, h in zip(lines, line_heights):
        bbox = font.getbbox(line)
        w = bbox[2] - bbox[0]
        x = (TARGET_W - w) // 2 - bbox[0]
        # Drop shadow
        draw.text(
            (x + 3, y + 4), line, font=font, fill=(0, 0, 0, 180),
            stroke_width=p["stroke"], stroke_fill=(0, 0, 0, 180),
        )
        # Main text
        draw.text(
            (x, y), line, font=font, fill=(255, 255, 255, 255),
            stroke_width=p["stroke"], stroke_fill=(0, 0, 0, 230),
        )
        y += h + line_spacing

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, "PNG")

    # Compute y position on the 1080x1920 canvas based on anchor
    anchor = p["y_anchor"]
    if anchor == "top":
        y_offset = 180
    elif anchor == "safe_top":
        y_offset = 240
    elif anchor == "upper_third":
        # ~28% from top — above most subject faces in vertical framing
        y_offset = int(TARGET_H * 0.20)
    elif anchor == "bottom":
        y_offset = TARGET_H - canvas_h - 360
    elif anchor == "watermark":
        y_offset = TARGET_H - canvas_h - 40
    else:  # center fallback
        y_offset = (TARGET_H - canvas_h) // 2

    return out_path, y_offset


def render_all_overlays(
    overlays: list[TextOverlay], out_dir: Path
) -> list[tuple[Path, int]]:
    """Render every overlay. Returns list of (png_path, y_offset)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, ov in enumerate(overlays):
        png_path = out_dir / f"overlay_{i:02d}.png"
        results.append(render_overlay_png(ov, png_path))
    return results
