"""Script alignment.

Indie creator writes a script. They record talking-head A-roll covering the
script (with improv). They also have B-roll. We align the script to actual
spoken words from Whisper transcripts, producing structured ScriptBeats with
timestamps, so the story stage can build an EDL that locks to script timing.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import BaseModel, Field

from montage.schemas import ClipAnalysis, SpeechSegment


_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")
_BEAT_BREAK = re.compile(r"\n\s*\n+")


class ScriptBeat(BaseModel):
    """One beat of a script aligned to source footage."""

    index: int
    text: str = Field(description="The script line / sentence for this beat")
    a_roll_asset_path: Path | None = Field(
        default=None, description="Best matching A-roll source clip"
    )
    a_roll_in_s: float | None = None
    a_roll_out_s: float | None = None
    match_score: float = Field(default=0.0, description="Similarity 0-1")


class Script(BaseModel):
    raw: str
    beats: list[ScriptBeat]


def load_script(path: Path) -> Script:
    """Load and parse a script file into ScriptBeats.

    Splits on double-newline (explicit beat breaks) first, then sentence
    boundaries. Empty lines and pure stage directions in [brackets] are
    ignored.
    """
    raw = path.read_text(encoding="utf-8").strip()
    # Drop bracketed stage directions
    cleaned = re.sub(r"\[[^\]]+\]", "", raw)
    # Split on double-newline OR on sentence boundary
    if _BEAT_BREAK.search(cleaned):
        chunks = _BEAT_BREAK.split(cleaned)
    else:
        chunks = _SENTENCE_BREAK.split(cleaned)
    beats: list[ScriptBeat] = []
    for i, chunk in enumerate(c.strip() for c in chunks):
        if not chunk:
            continue
        beats.append(ScriptBeat(index=len(beats), text=chunk))
    return Script(raw=raw, beats=beats)


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", " ", text.lower()).strip()


def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _best_window(
    script_line: str,
    transcript: list[SpeechSegment],
    min_window: float = 1.0,
    max_window: float = 12.0,
) -> tuple[float, float, float] | None:
    """Find the best (start_s, end_s, score) within transcript that matches script_line.

    Greedy: try every contiguous span of segments whose total duration falls
    in [min_window, max_window]. Score by similarity. Return best.
    """
    if not transcript:
        return None
    n = len(transcript)
    best: tuple[float, float, float] | None = None
    for i in range(n):
        joined = ""
        for j in range(i, n):
            joined = (joined + " " + transcript[j].text).strip()
            start = transcript[i].start_s
            end = transcript[j].end_s
            dur = end - start
            if dur < min_window:
                continue
            if dur > max_window:
                break
            score = _similarity(script_line, joined)
            if best is None or score > best[2]:
                best = (start, end, score)
    return best


def align_script(
    script: Script,
    speech_by_path: dict[Path, list[SpeechSegment]],
    analyses: list[ClipAnalysis],
    min_score: float = 0.35,
) -> Script:
    """For each ScriptBeat, find the best A-roll clip + window that matches.

    Returns a NEW Script with beat.a_roll_asset_path / in_s / out_s populated.
    """
    # Only consider A-roll clips when picking matches
    a_roll_paths = {a.asset_path for a in analyses if a.shot_type == "a_roll"}
    candidates = {
        p: segs for p, segs in speech_by_path.items() if p in a_roll_paths
    } or speech_by_path  # fall back to any clip with speech

    new_beats: list[ScriptBeat] = []
    for beat in script.beats:
        best_path: Path | None = None
        best_window: tuple[float, float, float] | None = None
        for path, transcript in candidates.items():
            w = _best_window(beat.text, transcript)
            if w is None:
                continue
            if best_window is None or w[2] > best_window[2]:
                best_window = w
                best_path = path
        if best_window and best_window[2] >= min_score:
            start, end, score = best_window
            new_beats.append(
                beat.model_copy(
                    update={
                        "a_roll_asset_path": best_path,
                        "a_roll_in_s": start,
                        "a_roll_out_s": end,
                        "match_score": score,
                    }
                )
            )
        else:
            new_beats.append(beat)  # un-matched, story stage can ad-lib

    return Script(raw=script.raw, beats=new_beats)


def format_for_prompt(script: Script) -> str:
    """Render the aligned script for the story-stage Opus prompt."""
    if not script.beats:
        return "(no script provided)"
    lines = []
    for b in script.beats:
        if b.a_roll_asset_path is not None:
            lines.append(
                f"BEAT {b.index} (matched A-roll {b.a_roll_asset_path.name} "
                f"[{b.a_roll_in_s:.1f}-{b.a_roll_out_s:.1f}] score={b.match_score:.2f}):\n"
                f"  \"{b.text}\""
            )
        else:
            lines.append(f"BEAT {b.index} (UNMATCHED — pick a fitting B-roll):\n  \"{b.text}\"")
    return "\n".join(lines)


def save(script: Script, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(script.model_dump_json(indent=2))


def load(path: Path) -> Script:
    return Script.model_validate_json(path.read_text())
