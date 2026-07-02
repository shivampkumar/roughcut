"""Reference reel analysis. Pro creators feed in 1-5 reels they love,
Gemini extracts a style descriptor paragraph that injects into the story prompt."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import ValidationError

from montage.schemas import RefStyle

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

_PROMPT = """You are analyzing 1-5 reference reels chosen by a content creator as exemplars of the style they want to emulate.

Output strict JSON matching this schema:

{
  "summary": "<one paragraph describing the shared aesthetic across all references. Pacing, energy, color palette, audio choices (music vs voiceover vs original audio), text overlay style, cut style (hard cuts vs crossfades vs whip pans), shot composition, lighting mood, overall emotional register>",
  "pacing": "slow | medium | fast | frenetic",
  "energy": "calm | medium | hype | unhinged",
  "cut_frequency_hz": 0.5,
  "typical_overlay_style": "bold_center | bottom_sub | top_caption | sparse | none"
}

Be specific. This descriptor will be fed to another model that builds an edit; vague summaries produce vague edits.

Output ONLY the JSON object. No prose, no markdown."""


def _client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) missing")
    return genai.Client(api_key=key)


async def _upload(client: genai.Client, path: Path):
    f = await asyncio.to_thread(client.files.upload, file=str(path))
    while f.state == "PROCESSING":
        await asyncio.sleep(2)
        f = await asyncio.to_thread(client.files.get, name=f.name)
    if f.state != "ACTIVE":
        raise RuntimeError(f"Gemini file state {f.state} for {path}")
    return f


async def analyze_async(paths: list[Path], model: str = DEFAULT_MODEL) -> RefStyle:
    if not paths:
        raise ValueError("No reference paths provided")
    client = _client()
    files = await asyncio.gather(*[_upload(client, p) for p in paths])
    resp = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=[*files, _PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    data = json.loads(resp.text)
    try:
        return RefStyle.model_validate(data)
    except ValidationError as e:
        raise RuntimeError(f"RefStyle parse failed: {e}\nRaw: {resp.text}") from e


def analyze(paths: list[Path], model: str = DEFAULT_MODEL) -> RefStyle:
    return asyncio.run(analyze_async(paths, model))


def save(style: RefStyle, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(style.model_dump_json(indent=2))


def load(path: Path) -> RefStyle:
    return RefStyle.model_validate_json(path.read_text())
