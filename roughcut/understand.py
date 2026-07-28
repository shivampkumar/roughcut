"""Per-clip understanding via Gemini.

For each MediaAsset, ask Gemini to score it on multiple axes + extract speech +
identify best moments. Returns ClipAnalysis objects.

Quality matters here. Prompt rubric is everything.
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

from google import genai
from google.genai import types
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress

from roughcut.schemas import ClipAnalysis, MediaAsset, SpeechSegment

console = Console()

# Switch via env: GEMINI_MODEL=gemini-2.5-pro or gemini-3.1-pro etc.
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")

_VIDEO_PROMPT = """You are a senior film editor analyzing raw footage for a 30-second vertical short. Be rigorous, honest, opinionated.

For the video provided, return strict JSON matching this schema:

{
  "asset_path": "<ignored, will be set by caller>",
  "timeline": [
    {"t": 0.0, "v": "framing/movement - what is on screen, max 15 words", "s": "speech at this moment or null"}
  ],
  "subjects": ["main person/object visible, then secondary subjects"],
  "action": "ONE sentence: what is actually happening, plainly",
  "emotion": "the dominant feeling (e.g. 'joyful chaos', 'quiet intimacy', 'pure hype', 'awe')",
  "setting": "where/when, one phrase (e.g. 'rooftop at golden hour', 'inside venue, dark with stage lights')",
  "composition_score": 0-10,    // framing, rule of thirds, depth, exposure
  "aesthetic_score": 0-10,      // colors, mood, would a stranger pause scrolling?
  "energy": 0-10,                // motion + emotional intensity. 0 = static slow, 10 = peak rage moment
  "narrative_role": "hook|setup|energy|climax|transition|close",
  "best_moments": [[start_s, end_s], ...],  // 1-3 best 1-3s windows in this clip, ranked best first. Use the timestamps you SEE.
  "motion_type": "static|pan|tracking|handheld|stable|zoom",
  "lighting": "natural|golden|harsh|dim|stage|neon|mixed",
  "shot_type": "a_roll|b_roll|performance|crowd|establishing|screen_rec|drone|selfie|still",   // SEE RULES below
  "has_speech": true/false,
  "speech_quality": 0-10,        // intelligibility. 0 = wind/noise only, 10 = crisp talk-to-camera
  "speech_segments": [
    {"start_s": float, "end_s": float, "text": "actual transcribed words", "confidence": 0-1}
  ],
  "speech_is_meaningful": true/false,   // Is it worth keeping? "haha hi" = false. "Today we drove 800 miles" = true.
  "has_music": true/false,       // diegetic music in the clip (concert audio, party speaker)
  "ambient_quality": 0-10,       // is the room/scene sound useful? (crowd cheer = high, fridge hum = low)
  "audio_notes": "one line on the audio. e.g. 'crowd roar peaks at 2.3s' or 'wind dominates, mute it'",
  "notes": "anything an editor should know. one line max."
}

TIMELINE RULES (the timeline is the source of truth; everything else summarizes it):
- Log significant visual changes only, max 12 events per clip. GROUP repeats: same shot with minor variation is ONE event with a count. A montage is ONE event.
- Every t is TOTAL SECONDS from clip start, within the clip's real duration. Never invent timestamps past the end.
- Mark camera-flips, selfie segments, blur/whip-pans, and darkness explicitly in v; downstream cutting decisions depend on these words.
- best_moments must each correspond to a region described by timeline events.

RULES:
- Be picky. Most clips are mediocre. Reserve 8+ scores for footage you'd actually post.
- best_moments timestamps refer to seconds within THIS clip starting at 0.
- Limit best_moments to 1-3 entries.
- For narrative_role: 'hook' = arresting first frame; 'climax' = peak emotional/visual; 'transition' = filler.
- speech_is_meaningful is strict — filler words, drunken yelling, indistinct chatter = false.
- shot_type taxonomy:
  - 'a_roll' = scripted/narrated talking head, creator addressing camera directly with meaningful speech. The "spine" of an indie YouTube video. is_a_roll requires: face visible, eye contact w/ lens, speech_is_meaningful=true.
  - 'b_roll' = illustration/cutaway footage (not narrated to camera). Default for most footage.
  - 'performance' = subject performing on stage (concert, dance, recital). NOT talking-head.
  - 'crowd' = audience reaction, group shots from audience POV.
  - 'establishing' = wide venue/location setup shot, no main subject yet.
  - 'screen_rec' = screen capture, app demo.
  - 'drone' = aerial/overhead.
  - 'selfie' = handheld facing self, casual not scripted.
  - 'still' = photo (only if input was an image, not video).
  Pick the SINGLE best-fitting tag. When unsure between a_roll and selfie, prefer a_roll only if speech is clearly scripted/narrated.
- Output ONLY the JSON object. No prose, no markdown fence.
"""

_PHOTO_PROMPT = """You are a senior photo editor scoring a still for use in a 30-second vertical short.

Return strict JSON:

{
  "asset_path": "<ignored>",
  "subjects": ["main subject", "secondary"],
  "action": "what's in the frame, one sentence",
  "emotion": "dominant feeling",
  "setting": "one phrase",
  "composition_score": 0-10,
  "aesthetic_score": 0-10,
  "energy": 0-10,
  "narrative_role": "hook|setup|energy|climax|transition|close",
  "best_moments": [],
  "motion_type": "static",
  "lighting": "...",
  "shot_type": "still",
  "has_speech": false,
  "speech_quality": 0,
  "speech_segments": [],
  "speech_is_meaningful": false,
  "has_music": false,
  "ambient_quality": 0,
  "audio_notes": "",
  "notes": "..."
}

Be picky. Output ONLY the JSON.
"""


def _client() -> genai.Client:
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY (or GOOGLE_API_KEY) missing from environment")
    return genai.Client(api_key=key)


async def _analyze_video(
    client: genai.Client,
    asset: MediaAsset,
    model: str,
    whisper_segments: list[SpeechSegment] | None = None,
) -> ClipAnalysis:
    # Upload file (Gemini Files API). For short clips < 20MB inline is also fine,
    # but uploading is the universal path.
    file = await asyncio.to_thread(
        client.files.upload, file=str(asset.path)
    )
    # Wait until ACTIVE
    while file.state == "PROCESSING":
        await asyncio.sleep(2)
        file = await asyncio.to_thread(client.files.get, name=file.name)
    if file.state != "ACTIVE":
        raise RuntimeError(f"Gemini file state {file.state} for {asset.path}")

    prompt = _VIDEO_PROMPT
    if whisper_segments:
        transcript = "\n".join(
            f"[{s.start_s:.2f}-{s.end_s:.2f}] \"{s.text}\""
            for s in whisper_segments
        )
        prompt = (
            f"GROUND TRUTH SPEECH TRANSCRIPT (from Whisper, authoritative):\n{transcript}\n\n"
            "Use these timestamps and text verbatim in speech_segments. Do not re-transcribe.\n"
            "Your job is to judge speech_is_meaningful and speech_quality given what you SEE and HEAR.\n\n"
            + _VIDEO_PROMPT
        )

    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=[file, prompt],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    data = json.loads(response.text)
    data["asset_path"] = str(asset.path)

    # Override Gemini's transcription with Whisper's if we have it
    if whisper_segments:
        data["speech_segments"] = [s.model_dump() for s in whisper_segments]
        data["has_speech"] = True
    return ClipAnalysis.model_validate(data)


async def _analyze_photo(client: genai.Client, asset: MediaAsset, model: str) -> ClipAnalysis:
    file = await asyncio.to_thread(client.files.upload, file=str(asset.path))
    response = await asyncio.to_thread(
        client.models.generate_content,
        model=model,
        contents=[file, _PHOTO_PROMPT],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.2,
        ),
    )
    data = json.loads(response.text)
    data["asset_path"] = str(asset.path)
    return ClipAnalysis.model_validate(data)


async def analyze_one(
    client: genai.Client,
    asset: MediaAsset,
    model: str,
    whisper_segments: list[SpeechSegment] | None = None,
    retries: int = 2,
) -> ClipAnalysis | None:
    """Analyze one asset. NEVER raises — one bad clip must not sink the batch."""
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            if asset.type == "video":
                return await _analyze_video(client, asset, model, whisper_segments=whisper_segments)
            return await _analyze_photo(client, asset, model)
        except Exception as e:  # API 5xx, JSON, validation, upload failures
            last_err = e
            if attempt < retries:
                await asyncio.sleep(2 * (attempt + 1))
    console.print(f"[red][skip] {asset.path.name}: {type(last_err).__name__}: {last_err}[/red]")
    return None


async def analyze_all(
    assets: list[MediaAsset],
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    speech_by_path: dict[Path, list[SpeechSegment]] | None = None,
) -> list[ClipAnalysis]:
    """Analyze every asset. Bounded concurrency to respect API rate limits."""
    client = _client()
    sem = asyncio.Semaphore(concurrency)
    results: list[ClipAnalysis] = []
    speech_by_path = speech_by_path or {}

    with Progress() as progress:
        task = progress.add_task("Analyzing", total=len(assets))

        async def _run(asset: MediaAsset) -> None:
            async with sem:
                segs = speech_by_path.get(asset.path)
                analysis = await analyze_one(client, asset, model, whisper_segments=segs)
                if analysis is not None:
                    results.append(analysis)
                progress.update(task, advance=1)

        await asyncio.gather(*[_run(a) for a in assets])

    return results


def analyze_all_sync(
    assets: list[MediaAsset],
    model: str = DEFAULT_MODEL,
    concurrency: int = 4,
    speech_by_path: dict[Path, list[SpeechSegment]] | None = None,
) -> list[ClipAnalysis]:
    return asyncio.run(analyze_all(assets, model, concurrency, speech_by_path))


def save(analyses: list[ClipAnalysis], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([a.model_dump(mode="json") for a in analyses], indent=2))


def load(path: Path) -> list[ClipAnalysis]:
    raw = json.loads(path.read_text())
    return [ClipAnalysis.model_validate(d) for d in raw]
