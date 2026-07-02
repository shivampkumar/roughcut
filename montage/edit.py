"""Conversational EDL editing.

User says: "at 0:08 add text 'WAIT FOR IT', change the SFX at 0:05 to a boom,
trim the close by 1 second".

Opus parses the instruction into structured EditOps. We apply them to the
current EDL and re-render. No full story re-run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Annotated, Literal, Union

from anthropic import Anthropic
from pydantic import BaseModel, Field, ValidationError

from montage.schemas import (
    EDL,
    EDLClip,
    SFXCue,
    SFXType,
    TextOverlay,
    OverlayType,
    AudioDecision,
    ClipAnalysis,
)

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")


# ---------- EditOp types (discriminated union) ----------


class AddOverlay(BaseModel):
    op: Literal["add_overlay"] = "add_overlay"
    text: str
    start_s: float
    duration_s: float = 2.0
    type: OverlayType = "caption"
    style: str = "bold_center"


class RemoveOverlay(BaseModel):
    """Remove overlay whose text contains the match string, or by index."""

    op: Literal["remove_overlay"] = "remove_overlay"
    match: str | None = None
    index: int | None = None


class UpdateOverlay(BaseModel):
    """Update an existing overlay matched by text substring or index."""

    op: Literal["update_overlay"] = "update_overlay"
    match: str | None = None
    index: int | None = None
    new_text: str | None = None
    new_start_s: float | None = None
    new_duration_s: float | None = None
    new_style: str | None = None


class AddSFX(BaseModel):
    op: Literal["add_sfx"] = "add_sfx"
    sfx: SFXType
    at_s: float
    volume_db: float = -6.0


class RemoveSFX(BaseModel):
    """Remove SFX cue near a timestamp (within ±0.5s) or by index."""

    op: Literal["remove_sfx"] = "remove_sfx"
    at_s: float | None = None
    index: int | None = None


class ReplaceSFX(BaseModel):
    op: Literal["replace_sfx"] = "replace_sfx"
    at_s: float
    new_sfx: SFXType
    new_volume_db: float | None = None


class TrimClip(BaseModel):
    """Trim clip by absolute new in/out OR by relative delta."""

    op: Literal["trim_clip"] = "trim_clip"
    index: int
    new_in_s: float | None = None
    new_out_s: float | None = None
    delta_in_s: float | None = None
    delta_out_s: float | None = None


class RemoveClip(BaseModel):
    op: Literal["remove_clip"] = "remove_clip"
    index: int


class SwapClip(BaseModel):
    """Replace clip at index with a new source segment."""

    op: Literal["swap_clip"] = "swap_clip"
    index: int
    new_asset_path: str
    new_in_s: float
    new_out_s: float
    new_crop_hint: str | None = None


class InsertClip(BaseModel):
    """Insert a new clip at given position (0 = before all)."""

    op: Literal["insert_clip"] = "insert_clip"
    position: int
    asset_path: str
    in_s: float
    out_s: float
    role: str = "transition"
    audio_decision: AudioDecision = "mute"


class UpdateClipAudio(BaseModel):
    op: Literal["update_clip_audio"] = "update_clip_audio"
    index: int
    audio_decision: AudioDecision | None = None
    gain_db: float | None = None


class UpdateMusic(BaseModel):
    op: Literal["update_music"] = "update_music"
    music_path: str | None = None
    gain_db: float | None = None
    ducks_under_speech: bool | None = None


EditOp = Annotated[
    Union[
        AddOverlay,
        RemoveOverlay,
        UpdateOverlay,
        AddSFX,
        RemoveSFX,
        ReplaceSFX,
        TrimClip,
        RemoveClip,
        SwapClip,
        InsertClip,
        UpdateClipAudio,
        UpdateMusic,
    ],
    Field(discriminator="op"),
]


class EditOpList(BaseModel):
    ops: list[EditOp]


# ---------- system prompt for the parser ----------

_PARSER_SYSTEM = """You translate natural-language video edit instructions into structured EditOps.

You receive:
- The current Edit Decision List (EDL) as JSON
- The original clip analyses (for swap_clip decisions)
- The user's free-form instruction

You output: a JSON object with an `ops` array containing one or more EditOp objects.

Each EditOp has an `op` discriminator field. Available op types:

- add_overlay {text, start_s, duration_s, type, style}
  - type: "hook" | "subtitle" | "caption" | "punchline" | "lower_third" | "closer"
  - style: "bold_center" | "bottom_sub" | "top_caption"
- remove_overlay {match | index}
- update_overlay {match | index, new_text, new_start_s, new_duration_s, new_style}
- add_sfx {sfx, at_s, volume_db}
  - sfx: "whoosh" | "impact" | "pop" | "shimmer" | "click" | "boom" | "swell" | "riser"
- remove_sfx {at_s | index}
- replace_sfx {at_s, new_sfx, new_volume_db}
- trim_clip {index, new_in_s OR delta_in_s, new_out_s OR delta_out_s}
- remove_clip {index}
- swap_clip {index, new_asset_path, new_in_s, new_out_s, new_crop_hint}
- insert_clip {position, asset_path, in_s, out_s, role, audio_decision}
- update_clip_audio {index, audio_decision, gain_db}
  - audio_decision: "keep_loud" | "keep_natural" | "duck" | "mute" | "subtitle_only"
- update_music {music_path, gain_db, ducks_under_speech}

RULES:
1. Timestamps in the user's instruction refer to the TIMELINE of the reel, not source clips.
2. Be conservative. Output the smallest number of ops that achieves the user's intent.
3. If the instruction is ambiguous, prefer the most likely interpretation given the EDL.
4. Use `index` when the user references a specific clip by number ("clip 3"). Use `at_s` when they reference a timestamp ("at 0:08").
5. For swap_clip and insert_clip, only use asset_paths that exist in the original analyses.
6. NEVER use em dashes in any text fields. Use commas, ellipses, or new sentences.

Output STRICT JSON: {"ops": [...]}. No prose. No markdown fences."""


# ---------- parser ----------


def parse_edits(
    instruction: str,
    edl: EDL,
    analyses: list[ClipAnalysis],
    model: str = DEFAULT_MODEL,
) -> list[EditOp]:
    """Use Opus to turn a natural-language instruction into structured EditOps."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from environment")
    client = Anthropic(api_key=api_key)

    edl_json = edl.model_dump_json(indent=2)
    available = "\n".join(
        f"- {a.asset_path} | {a.action} | best_moments={a.best_moments[:3]} | "
        f"aesthetic={a.aesthetic_score:.0f} energy={a.energy:.0f}"
        for a in analyses
    )

    user = (
        f"Current EDL (timeline reads top-to-bottom):\n```json\n{edl_json}\n```\n\n"
        f"Available source clips:\n{available}\n\n"
        f"User instruction:\n{instruction}\n\n"
        "Output the JSON ops object now."
    )

    msg = client.messages.create(
        model=model,
        max_tokens=2048,
        system=_PARSER_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(b.text for b in msg.content if hasattr(b, "text")).strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()

    try:
        parsed = EditOpList.model_validate_json(text)
    except ValidationError as e:
        raise RuntimeError(
            f"Could not parse edit ops:\n{e}\n\nRaw output:\n{text}"
        ) from e
    return parsed.ops


# ---------- appliers ----------


def _find_overlay_index(
    edl: EDL, match: str | None, index: int | None
) -> int | None:
    if index is not None and 0 <= index < len(edl.overlays):
        return index
    if match:
        match_low = match.lower()
        for i, ov in enumerate(edl.overlays):
            if match_low in ov.text.lower():
                return i
    return None


def _find_sfx_index(edl: EDL, at_s: float | None, index: int | None) -> int | None:
    if index is not None and 0 <= index < len(edl.sfx):
        return index
    if at_s is not None:
        closest = min(
            range(len(edl.sfx)),
            key=lambda i: abs(edl.sfx[i].at_s - at_s),
            default=None,
        )
        if closest is not None and abs(edl.sfx[closest].at_s - at_s) <= 0.5:
            return closest
    return None


def apply_one(edl: EDL, op) -> EDL:
    """Return a NEW EDL with the op applied. Mutates nothing."""
    arc = list(edl.arc)
    overlays = list(edl.overlays)
    sfx = list(edl.sfx)

    if isinstance(op, AddOverlay):
        overlays.append(
            TextOverlay(
                type=op.type,
                text=op.text,
                start_s=op.start_s,
                duration_s=op.duration_s,
                style=op.style,
            )
        )
    elif isinstance(op, RemoveOverlay):
        idx = _find_overlay_index(edl, op.match, op.index)
        if idx is not None:
            overlays.pop(idx)
    elif isinstance(op, UpdateOverlay):
        idx = _find_overlay_index(edl, op.match, op.index)
        if idx is not None:
            ov = overlays[idx]
            overlays[idx] = ov.model_copy(
                update={
                    k: v
                    for k, v in {
                        "text": op.new_text,
                        "start_s": op.new_start_s,
                        "duration_s": op.new_duration_s,
                        "style": op.new_style,
                    }.items()
                    if v is not None
                }
            )
    elif isinstance(op, AddSFX):
        sfx.append(SFXCue(sfx=op.sfx, at_s=op.at_s, volume_db=op.volume_db))
        sfx.sort(key=lambda c: c.at_s)
    elif isinstance(op, RemoveSFX):
        idx = _find_sfx_index(edl, op.at_s, op.index)
        if idx is not None:
            sfx.pop(idx)
    elif isinstance(op, ReplaceSFX):
        idx = _find_sfx_index(edl, op.at_s, None)
        if idx is not None:
            cue = sfx[idx]
            sfx[idx] = cue.model_copy(
                update={
                    "sfx": op.new_sfx,
                    **(
                        {"volume_db": op.new_volume_db}
                        if op.new_volume_db is not None
                        else {}
                    ),
                }
            )
    elif isinstance(op, TrimClip):
        if 0 <= op.index < len(arc):
            clip = arc[op.index]
            new_in = op.new_in_s if op.new_in_s is not None else clip.in_s + (op.delta_in_s or 0)
            new_out = (
                op.new_out_s if op.new_out_s is not None else clip.out_s + (op.delta_out_s or 0)
            )
            new_in = max(0.0, new_in)
            new_out = max(new_in + 0.3, new_out)
            arc[op.index] = clip.model_copy(update={"in_s": new_in, "out_s": new_out})
    elif isinstance(op, RemoveClip):
        if 0 <= op.index < len(arc):
            arc.pop(op.index)
    elif isinstance(op, SwapClip):
        if 0 <= op.index < len(arc):
            clip = arc[op.index]
            arc[op.index] = clip.model_copy(
                update={
                    "asset_path": Path(op.new_asset_path),
                    "in_s": op.new_in_s,
                    "out_s": op.new_out_s,
                    **(
                        {"crop_hint": op.new_crop_hint}
                        if op.new_crop_hint is not None
                        else {}
                    ),
                }
            )
    elif isinstance(op, InsertClip):
        pos = max(0, min(op.position, len(arc)))
        arc.insert(
            pos,
            EDLClip(
                asset_path=Path(op.asset_path),
                in_s=op.in_s,
                out_s=op.out_s,
                role=op.role,
                audio_decision=op.audio_decision,
            ),
        )
    elif isinstance(op, UpdateClipAudio):
        if 0 <= op.index < len(arc):
            clip = arc[op.index]
            arc[op.index] = clip.model_copy(
                update={
                    k: v
                    for k, v in {
                        "audio_decision": op.audio_decision,
                        "original_audio_gain_db": op.gain_db,
                    }.items()
                    if v is not None
                }
            )
    elif isinstance(op, UpdateMusic):
        updates = {}
        if op.music_path is not None:
            updates["music_path"] = Path(op.music_path)
        if op.gain_db is not None:
            updates["music_gain_db"] = op.gain_db
        if op.ducks_under_speech is not None:
            updates["music_ducks_under_speech"] = op.ducks_under_speech
        if updates:
            return edl.model_copy(update=updates)
    else:
        raise ValueError(f"Unknown op: {op}")

    return edl.model_copy(update={"arc": arc, "overlays": overlays, "sfx": sfx})


def apply_edits(edl: EDL, ops: list[EditOp]) -> EDL:
    """Apply ops in order; each returns a new EDL."""
    for op in ops:
        edl = apply_one(edl, op)
    return edl


def describe_op(op) -> str:
    """Human-readable summary for logging."""
    if isinstance(op, AddOverlay):
        return f"add overlay '{op.text}' @ {op.start_s:.1f}s for {op.duration_s:.1f}s"
    if isinstance(op, RemoveOverlay):
        target = f"index={op.index}" if op.index is not None else f"matching '{op.match}'"
        return f"remove overlay ({target})"
    if isinstance(op, UpdateOverlay):
        target = f"index={op.index}" if op.index is not None else f"matching '{op.match}'"
        return f"update overlay ({target}) → text={op.new_text!r}"
    if isinstance(op, AddSFX):
        return f"add SFX {op.sfx} @ {op.at_s:.1f}s ({op.volume_db}dB)"
    if isinstance(op, RemoveSFX):
        return f"remove SFX @ {op.at_s or op.index}"
    if isinstance(op, ReplaceSFX):
        return f"replace SFX @ {op.at_s:.1f}s → {op.new_sfx}"
    if isinstance(op, TrimClip):
        return f"trim clip {op.index} → in={op.new_in_s} out={op.new_out_s} Δin={op.delta_in_s} Δout={op.delta_out_s}"
    if isinstance(op, RemoveClip):
        return f"remove clip {op.index}"
    if isinstance(op, SwapClip):
        return f"swap clip {op.index} → {Path(op.new_asset_path).name}[{op.new_in_s:.1f}-{op.new_out_s:.1f}]"
    if isinstance(op, InsertClip):
        return f"insert clip @ {op.position}: {Path(op.asset_path).name}[{op.in_s:.1f}-{op.out_s:.1f}]"
    if isinstance(op, UpdateClipAudio):
        return f"clip {op.index} audio → {op.audio_decision} gain={op.gain_db}dB"
    if isinstance(op, UpdateMusic):
        return f"music → {op.music_path} gain={op.gain_db}dB"
    return str(op)
