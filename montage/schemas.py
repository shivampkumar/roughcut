from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

MediaType = Literal["photo", "video"]
NarrativeRole = Literal["hook", "setup", "energy", "climax", "transition", "close"]
MotionType = Literal["static", "pan", "tracking", "handheld", "stable", "zoom"]

# Shot type — drives A-roll/B-roll decisions in story stage
ShotType = Literal[
    "a_roll",         # talking head, creator narrating to camera
    "b_roll",         # illustration, cutaways, generic atmosphere
    "performance",    # subject performing on stage (concert, recital, dance)
    "crowd",          # audience reactions, group shots
    "establishing",   # wide setup shot of venue/location
    "screen_rec",     # screen recording, demo footage
    "drone",          # aerial
    "selfie",         # camera held by subject, addressing
    "still",          # photo (not video)
]

# Per-clip audio decisions — how to handle original audio
AudioDecision = Literal[
    "keep_loud",          # original voice/speech is the point; music ducks under
    "keep_natural",       # ambient sound stays at moderate level, music mid
    "duck",               # original audio low, music dominant
    "mute",               # drop original entirely, music only
    "subtitle_only",      # mute original but transcribe speech to subtitles
]

# Types of text on screen
OverlayType = Literal[
    "hook",         # big opening text card ("the night we met...")
    "subtitle",     # transcribed speech, bottom-third
    "caption",      # descriptive bottom text
    "punchline",    # mid-clip reveal text
    "lower_third",  # name/location label
    "closer",       # outro card
]

# SFX library tags — abstract, mapped to actual sound files at assembly
SFXType = Literal[
    "whoosh",       # cut/transition
    "impact",       # beat hit
    "pop",          # text appear
    "shimmer",      # photo zoom
    "click",        # text snap
    "boom",         # heavy emphasis
    "swell",        # build before drop
    "riser",        # tension build
]


class MediaAsset(BaseModel):
    path: Path
    type: MediaType
    duration_s: float = 0.0
    width: int
    height: int
    fps: float | None = None
    timestamp: datetime | None = None
    gps: tuple[float, float] | None = None
    has_audio: bool = False
    audio_rms_db: float | None = Field(default=None, description="Average loudness")

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 1.0


class SpeechSegment(BaseModel):
    start_s: float
    end_s: float
    text: str
    confidence: float = 1.0


class ClipAnalysis(BaseModel):
    asset_path: Path

    # Visual content
    subjects: list[str] = Field(default_factory=list, description="People/objects visible")
    action: str = Field(description="What is happening, one sentence")
    emotion: str = Field(description="Dominant emotion/vibe")
    setting: str = Field(description="Where/when, one phrase")

    # Quality scores
    composition_score: float = Field(ge=0, le=10, description="Cinematography quality")
    aesthetic_score: float = Field(ge=0, le=10, description="Visual appeal")
    energy: float = Field(ge=0, le=10, description="Motion + emotional intensity")

    # Narrative
    narrative_role: NarrativeRole
    best_moments: list[tuple[float, float]] = Field(
        default_factory=list, description="(start_s, end_s) — best 1-3s segments"
    )

    # Motion/camera
    motion_type: MotionType = "stable"
    lighting: str = "natural"

    # Shot taxonomy — drives A-roll/B-roll decisions
    shot_type: ShotType = "b_roll"

    @field_validator("motion_type", mode="before")
    @classmethod
    def _coerce_motion(cls, v: str) -> str:
        valid = {"static", "pan", "tracking", "handheld", "stable", "zoom"}
        if isinstance(v, str):
            v_low = v.lower().strip()
            if v_low in valid:
                return v_low
            # Common aliases
            aliases = {
                "still": "static", "steady": "stable", "shake": "handheld",
                "shaky": "handheld", "panning": "pan", "track": "tracking",
                "zooming": "zoom", "fixed": "static",
            }
            return aliases.get(v_low, "stable")
        return "stable"

    # Audio analysis — drives audio_decision in story stage
    has_speech: bool = False
    speech_quality: float = Field(default=0.0, ge=0, le=10, description="Clarity, intelligibility")
    speech_segments: list[SpeechSegment] = Field(default_factory=list)
    speech_is_meaningful: bool = Field(
        default=False,
        description="Is the speech worth keeping/subtitling? (vs filler 'haha', wind noise)",
    )
    has_music: bool = False  # source music — concert, party background
    ambient_quality: float = Field(
        default=0.0, ge=0, le=10, description="Is the ambient sound diegetic/useful?"
    )
    audio_notes: str = ""

    notes: str = ""


class TextOverlay(BaseModel):
    """One text element on screen."""

    type: OverlayType
    text: str
    start_s: float = Field(description="Time within the final reel, not source clip")
    duration_s: float = 2.0
    style: str = Field(default="bold_center", description="Style preset name")


class SFXCue(BaseModel):
    """One sound effect cue."""

    sfx: SFXType
    at_s: float = Field(description="Time within final reel")
    volume_db: float = -6.0


class EDLClip(BaseModel):
    """One clip in the final cut — knows its own audio mix decision."""

    asset_path: Path
    in_s: float = 0.0
    out_s: float = 1.0
    crop_hint: str = "center"
    role: NarrativeRole = "transition"

    # Motion + transition vocabulary
    transition_in: Literal["cut", "fade"] = "cut"
    punch_in: bool = Field(
        default=False, description="Slow zoom-in over the clip. For held/static shots."
    )

    # Audio mix decision for THIS clip's original audio
    audio_decision: AudioDecision = "mute"
    original_audio_gain_db: float = Field(
        default=0.0, description="dB adjustment to original track (negative = quieter)"
    )

    @property
    def duration_s(self) -> float:
        return self.out_s - self.in_s


class BrandKit(BaseModel):
    """Brand styling for ICP 2 (creators). Stored as JSON, loaded per project."""

    name: str = "default"
    logo_path: Path | None = None
    logo_position: Literal["top_left", "top_right", "bottom_left", "bottom_right"] = "top_right"
    logo_opacity: float = Field(default=0.85, ge=0, le=1)
    logo_size_pct: float = Field(default=8.0, ge=1, le=30, description="% of frame width")
    font_primary_path: Path | None = None
    color_primary: str = "#FFFFFF"
    color_accent: str = "#FFD700"
    lut_path: Path | None = Field(default=None, description=".cube color LUT")
    watermark_text: str | None = None
    voice_id: str | None = Field(default=None, description="ElevenLabs voice id for VO")
    intro_sting_path: Path | None = None
    outro_sting_path: Path | None = None


class RefStyle(BaseModel):
    """Style descriptor extracted from reference reels."""

    summary: str = Field(description="One-paragraph style note: pacing, energy, palette, audio, text style")
    pacing: str = "medium"
    energy: str = "medium"
    cut_frequency_hz: float = 0.5
    typical_overlay_style: str = "bold_center"


class EDL(BaseModel):
    """Edit Decision List — the output of story stage."""

    event_type: str = Field(description="concert, trip, wedding, party, casual, etc.")
    target_duration_s: float = 30.0

    # Visual sequence
    arc: list[EDLClip]

    # Music bed
    music_mood: str
    music_genre: str
    music_bpm: int = 120
    music_path: Path | None = None
    music_gain_db: float = -10.0  # under voices, over silent clips
    music_ducks_under_speech: bool = True

    # Text overlays — placed on the timeline, not on individual clips
    overlays: list[TextOverlay] = Field(default_factory=list)

    # Sound effects punctuation
    sfx: list[SFXCue] = Field(default_factory=list)

    # Voiceover (optional) — when no clip speech carries the story
    voiceover_script: str | None = None
    voiceover_voice_id: str | None = None  # ElevenLabs voice id

    # Upload metadata
    upload_caption: str | None = None
    hashtags: list[str] = Field(default_factory=list)

    @property
    def total_duration_s(self) -> float:
        return sum(c.duration_s for c in self.arc)
