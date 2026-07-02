"""Whisper-based speech transcription. Ground-truth speech segments per clip.

Uses mlx-whisper (Apple Silicon native). Falls back to no-op if mlx-whisper
isn't installed (e.g., on Linux).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from roughcut.ingest import extract_audio
from roughcut.schemas import MediaAsset, SpeechSegment

DEFAULT_MODEL = os.getenv("WHISPER_MODEL", "mlx-community/whisper-large-v3-turbo")
MIN_SEGMENT_DURATION = 0.3   # drop transcripts shorter than this
MIN_TEXT_LENGTH = 2          # drop "uh", "."


def _load_mlx_whisper():
    try:
        import mlx_whisper  # type: ignore
        return mlx_whisper
    except ImportError:
        return None


def transcribe_audio(
    audio_path: Path, model: str = DEFAULT_MODEL
) -> list[SpeechSegment]:
    """Transcribe a 16kHz mono wav. Returns SpeechSegments."""
    mlx_whisper = _load_mlx_whisper()
    if mlx_whisper is None:
        return []

    result = mlx_whisper.transcribe(
        str(audio_path),
        path_or_hf_repo=model,
        verbose=False,
    )
    segments: list[SpeechSegment] = []
    for s in result.get("segments", []):
        start = float(s.get("start", 0.0))
        end = float(s.get("end", start))
        text = (s.get("text") or "").strip()
        if end - start < MIN_SEGMENT_DURATION:
            continue
        if len(text) < MIN_TEXT_LENGTH:
            continue
        avg_logprob = float(s.get("avg_logprob", 0.0))
        # Convert log-prob to a pseudo-confidence in [0,1]
        confidence = max(0.0, min(1.0, 1.0 + avg_logprob / 2))
        segments.append(
            SpeechSegment(start_s=start, end_s=end, text=text, confidence=confidence)
        )
    return segments


def transcribe_video(video: MediaAsset, work_dir: Path | None = None) -> list[SpeechSegment]:
    """Extract audio + transcribe. Returns [] if no audio or whisper unavailable."""
    if not video.has_audio:
        return []
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="roughcut_speech_"))
    audio_path = work_dir / f"{video.path.stem}.wav"
    if not audio_path.exists():
        if extract_audio(video, audio_path) is None:
            return []
    return transcribe_audio(audio_path)


def transcribe_all(
    videos: list[MediaAsset], work_dir: Path | None = None
) -> dict[Path, list[SpeechSegment]]:
    """Batch over multiple videos. Sequential — Whisper is GPU-bound."""
    out: dict[Path, list[SpeechSegment]] = {}
    for v in videos:
        if v.type != "video" or not v.has_audio:
            continue
        out[v.path] = transcribe_video(v, work_dir=work_dir)
    return out
