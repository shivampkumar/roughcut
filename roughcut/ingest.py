"""Walk a folder, probe media files, return MediaAsset objects.

Also: helpers to extract frames + audio for downstream stages.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path

from PIL import Image, ExifTags

from roughcut.schemas import MediaAsset

VIDEO_EXT = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
PHOTO_EXT = {".jpg", ".jpeg", ".png", ".heic", ".webp"}


def walk_media(root: Path) -> list[Path]:
    """Return sorted list of media files under root (recursive)."""
    root = root.expanduser().resolve()
    paths: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in (VIDEO_EXT | PHOTO_EXT):
            paths.append(p)
    return sorted(paths)


def _ffprobe(path: Path) -> dict:
    """Run ffprobe, return parsed JSON."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr}")
    return json.loads(result.stdout)


def _video_audio_rms(path: Path) -> float | None:
    """Mean RMS in dBFS over the whole audio track. None if no audio."""
    result = subprocess.run(
        [
            "ffmpeg",
            "-nostats",
            "-i", str(path),
            "-af", "astats=metadata=1:reset=0",
            "-f", "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    # Parse "Overall.RMS_level: -23.45" from stderr
    for line in result.stderr.splitlines():
        if "Overall.RMS_level" in line or "RMS level dB" in line:
            try:
                return float(line.split(":")[-1].strip())
            except ValueError:
                continue
    return None


def _rotation(video_stream: dict) -> int:
    """Return display rotation in degrees (-180..180)."""
    # Modern ffmpeg writes rotation via side_data_list with display matrix
    for sd in video_stream.get("side_data_list", []):
        if "rotation" in sd:
            try:
                return int(round(float(sd["rotation"])))
            except (TypeError, ValueError):
                pass
    # Older form: tags.rotate
    tags = video_stream.get("tags") or {}
    rot = tags.get("rotate")
    if rot is not None:
        try:
            return int(rot)
        except ValueError:
            return 0
    return 0


def probe_video(path: Path) -> MediaAsset:
    """Extract MediaAsset for a video file."""
    info = _ffprobe(path)
    video_stream = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    audio_stream = next((s for s in info["streams"] if s["codec_type"] == "audio"), None)
    if video_stream is None:
        raise ValueError(f"No video stream in {path}")

    duration = float(info["format"].get("duration", 0))
    width = int(video_stream["width"])
    height = int(video_stream["height"])
    # Honor display rotation: if rotated ±90, the effective frame is swapped.
    rotation = _rotation(video_stream)
    if abs(rotation) == 90 or abs(rotation) == 270:
        width, height = height, width
    fps_str = video_stream.get("r_frame_rate", "0/1")
    num, den = fps_str.split("/")
    fps = float(num) / float(den) if float(den) > 0 else None

    timestamp = None
    creation = info["format"].get("tags", {}).get("creation_time")
    if creation:
        try:
            dt = datetime.fromisoformat(creation.replace("Z", "+00:00"))
            timestamp = dt.replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            pass

    rms_db = _video_audio_rms(path) if audio_stream else None

    return MediaAsset(
        path=path,
        type="video",
        duration_s=duration,
        width=width,
        height=height,
        fps=fps,
        timestamp=timestamp,
        has_audio=audio_stream is not None,
        audio_rms_db=rms_db,
    )


def probe_photo(path: Path) -> MediaAsset:
    """Extract MediaAsset for a still photo."""
    with Image.open(path) as img:
        width, height = img.size
        exif = img.getexif() if hasattr(img, "getexif") else None

    timestamp = None
    gps: tuple[float, float] | None = None
    if exif:
        tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
        dt_str = tags.get("DateTimeOriginal") or tags.get("DateTime")
        if dt_str:
            try:
                timestamp = datetime.strptime(dt_str, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass
        # GPS extraction kept minimal — TODO: full ExifTags.GPSTAGS parse
    return MediaAsset(
        path=path,
        type="photo",
        duration_s=0.0,
        width=width,
        height=height,
        timestamp=timestamp,
        gps=gps,
        has_audio=False,
    )


def probe(path: Path) -> MediaAsset:
    ext = path.suffix.lower()
    if ext in VIDEO_EXT:
        return probe_video(path)
    if ext in PHOTO_EXT:
        return probe_photo(path)
    raise ValueError(f"Unsupported file type: {path}")


def ingest_folder(root: Path) -> list[MediaAsset]:
    """Walk + probe every supported file under root."""
    assets = []
    for p in walk_media(root):
        try:
            assets.append(probe(p))
        except Exception as e:
            print(f"[skip] {p.name}: {e}")
    # Sort by capture timestamp when available, then filename
    assets.sort(key=lambda a: (a.timestamp or datetime.max, a.path.name))
    return assets


# ---------- frame + audio extraction helpers ----------

def extract_frames(
    video: MediaAsset, out_dir: Path, fps: float = 1.0
) -> list[Path]:
    """Sample frames at `fps` fps. Returns list of frame paths.

    Used to feed Gemini for video understanding when full-video upload is too big.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = video.path.stem
    pattern = out_dir / f"{stem}_%04d.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video.path),
            "-vf", f"fps={fps}",
            "-q:v", "3",
            str(pattern),
        ],
        check=True,
        capture_output=True,
    )
    return sorted(out_dir.glob(f"{stem}_*.jpg"))


def extract_audio(video: MediaAsset, out_path: Path) -> Path | None:
    """Pull audio to WAV. Returns path, or None if no audio."""
    if not video.has_audio:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i", str(video.path),
            "-ac", "1",            # mono
            "-ar", "16000",        # 16kHz for whisper compat
            "-vn",
            str(out_path),
        ],
        check=True,
        capture_output=True,
    )
    return out_path
