"""Music + audio synthesis.

- analyze_track: beat detection, BPM, downbeats via librosa
- snap_edl_to_beats: adjust EDLClip cut points so they land on musical beats
- generate_voiceover (optional): ElevenLabs TTS for EDL.voiceover_script
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np

from montage.schemas import EDL, EDLClip


@dataclass
class TrackAnalysis:
    path: Path
    duration_s: float
    bpm: float
    beat_times: list[float]
    downbeat_times: list[float]
    energy_curve: list[float]  # 1-second buckets

    def nearest_beat(self, t: float) -> float:
        if not self.beat_times:
            return t
        return min(self.beat_times, key=lambda b: abs(b - t))


def analyze_track(path: Path) -> TrackAnalysis:
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    # Tempo + beats
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr, units="time")
    beat_times = beat_frames.tolist() if hasattr(beat_frames, "tolist") else list(beat_frames)

    # Downbeats: take every 4th beat as rough heuristic; good enough for cut decisions
    downbeats = beat_times[::4] if beat_times else []

    # Energy curve (RMS in 1s buckets)
    hop = sr  # 1 second
    rms = librosa.feature.rms(y=y, frame_length=hop * 2, hop_length=hop)[0]
    energy_curve = rms.tolist()

    return TrackAnalysis(
        path=path,
        duration_s=float(duration),
        bpm=float(tempo) if not hasattr(tempo, "item") else float(tempo.item()),
        beat_times=beat_times,
        downbeat_times=downbeats,
        energy_curve=energy_curve,
    )


def snap_edl_to_beats(
    edl: EDL,
    track: TrackAnalysis,
    snap_strength: float = 0.7,
) -> EDL:
    """Adjust EDLClip durations so cut boundaries align with beats.

    snap_strength: 0 = ignore beats; 1 = force every cut to nearest beat.
    Preserves total target duration approximately.
    """
    if not track.beat_times:
        return edl

    new_arc: list[EDLClip] = []
    timeline_t = 0.0
    for clip in edl.arc:
        # The end time on the final timeline
        nominal_end = timeline_t + clip.duration_s
        snapped_end = track.nearest_beat(nominal_end)
        # Lerp between nominal and snapped by snap_strength
        target_end = nominal_end + (snapped_end - nominal_end) * snap_strength
        new_dur = max(0.3, target_end - timeline_t)  # never shorter than 0.3s

        # Shift clip in/out by the delta, keeping in_s anchored
        new_clip = clip.model_copy(update={"out_s": clip.in_s + new_dur})
        new_arc.append(new_clip)
        timeline_t += new_dur

    return edl.model_copy(update={"arc": new_arc})


# ---------- voiceover (optional) ----------


def generate_voiceover(
    text: str, out_path: Path, voice_id: str | None = None
) -> Path | None:
    """ElevenLabs TTS to WAV. Returns path, or None if no key configured."""
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        return None
    from elevenlabs.client import ElevenLabs

    client = ElevenLabs(api_key=api_key)
    voice_id = voice_id or os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")
    audio = client.text_to_speech.convert(
        voice_id=voice_id,
        text=text,
        model_id="eleven_multilingual_v2",
        output_format="mp3_44100_128",
    )
    mp3_tmp = out_path.with_suffix(".mp3")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mp3_tmp, "wb") as f:
        for chunk in audio:
            f.write(chunk)
    # Normalize to wav for ffmpeg downstream consistency
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(mp3_tmp), "-ac", "1", "-ar", "44100", str(out_path)],
        check=True,
        capture_output=True,
    )
    mp3_tmp.unlink(missing_ok=True)
    return out_path


# ---------- background SFX library (stub) ----------

_SFX_LIBRARY_DIR = Path(os.getenv("MONTAGE_SFX_DIR", str(Path(__file__).parent / "assets" / "sfx")))


def sfx_path(sfx_name: str) -> Path | None:
    """Return path to a packaged SFX wav, or None if missing."""
    candidate = _SFX_LIBRARY_DIR / f"{sfx_name}.wav"
    return candidate if candidate.exists() else None
