"""Generate a synthetic 'event folder' to test the full pipeline without real footage.

Produces in /tmp/montage_fixtures:
  - aroll_speech.mp4    720x1280 portrait, real TTS speech (macOS say) — tests Whisper,
                        a_roll detection, script alignment
  - clip_rotated.mp4    1280x720 stored landscape + rotate=-90 metadata — tests rotation swap
  - clip_music.mp4      testsrc2 with chord audio — tests has_music path
  - clip_silent.mp4     video only, no audio stream — tests silent-audio handling
  - photo_wide.jpg      4000x2250 — tests horizontal crop + Ken Burns
  - photo_tall.jpg      1080x2400 — tests vertical crop
  - script.txt          matches the say speech — tests alignment end-to-end
"""

import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

OUT = Path("/tmp/montage_fixtures")
OUT.mkdir(parents=True, exist_ok=True)

SPEECH = (
    "Welcome back to the channel. Today we are testing the montage pipeline. "
    "This synthetic clip proves that script alignment works end to end."
)


def run(cmd: list[str]) -> None:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:6])}... failed:\n{r.stderr[-800:]}")


def make_speech_clip() -> None:
    aiff = OUT / "speech.aiff"
    run(["say", "-o", str(aiff), SPEECH])
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=size=720x1280:rate=30",
        "-i", str(aiff),
        "-shortest",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-ar", "44100", "-ac", "2",
        str(OUT / "aroll_speech.mp4"),
    ])
    aiff.unlink(missing_ok=True)


def make_rotated_clip() -> None:
    tmp = OUT / "_rot_tmp.mp4"
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "smptebars=size=1280x720:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=5",
        "-t", "5",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(tmp),
    ])
    # mp4 muxer ignores -metadata rotate; write a real display matrix instead.
    run([
        "ffmpeg", "-y", "-display_rotation", "-90", "-i", str(tmp),
        "-c", "copy", str(OUT / "clip_rotated.mp4"),
    ])
    tmp.unlink(missing_ok=True)


def make_music_clip() -> None:
    # Chord-ish audio: three sines mixed
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "mandelbrot=size=720x1280:rate=30",
        "-f", "lavfi",
        "-i", "sine=frequency=261.63:duration=6",
        "-f", "lavfi",
        "-i", "sine=frequency=329.63:duration=6",
        "-f", "lavfi",
        "-i", "sine=frequency=392:duration=6",
        "-filter_complex", "[1:a][2:a][3:a]amix=inputs=3[aout]",
        "-map", "0:v", "-map", "[aout]",
        "-t", "6",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        str(OUT / "clip_music.mp4"),
    ])


def make_silent_clip() -> None:
    run([
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=size=720x1280:rate=30",
        "-t", "4", "-an",
        "-c:v", "libx264", "-preset", "veryfast", "-pix_fmt", "yuv420p",
        str(OUT / "clip_silent.mp4"),
    ])


def make_photos() -> None:
    # Wide photo with an off-center bright subject (tests crop anchoring)
    img = Image.new("RGB", (4000, 2250))
    d = ImageDraw.Draw(img)
    for x in range(0, 4000, 8):
        c = int(40 + 60 * (x / 4000))
        d.rectangle([x, 0, x + 8, 2250], fill=(c, c // 2, c))
    d.ellipse([2600, 700, 3400, 1500], fill=(250, 220, 80))
    img.save(OUT / "photo_wide.jpg", quality=90)

    img2 = Image.new("RGB", (1080, 2400))
    d2 = ImageDraw.Draw(img2)
    for y in range(0, 2400, 8):
        c = int(30 + 80 * (y / 2400))
        d2.rectangle([0, y, 1080, y + 8], fill=(c // 2, c, c))
    d2.ellipse([340, 900, 740, 1300], fill=(240, 90, 90))
    img2.save(OUT / "photo_tall.jpg", quality=90)


def make_script() -> None:
    (OUT / "script.txt").write_text(
        "Welcome back to the channel.\n\n"
        "Today we are testing the montage pipeline.\n\n"
        "This synthetic clip proves that script alignment works end to end.\n"
    )


if __name__ == "__main__":
    make_speech_clip()
    make_rotated_clip()
    make_music_clip()
    make_silent_clip()
    make_photos()
    make_script()
    for p in sorted(OUT.iterdir()):
        print(p)
