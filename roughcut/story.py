"""Story stage. Claude Opus 4.7 turns clip analyses into an Edit Decision List.

This is the intelligence layer. The prompt is the product.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from anthropic import Anthropic
from pydantic import ValidationError

from roughcut.schemas import BrandKit, ClipAnalysis, EDL, MediaAsset, RefStyle
from roughcut.script import Script, format_for_prompt as _format_script

DEFAULT_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7")

_SYSTEM_CORE = """You are a senior short-form video editor. You make 20-40 second vertical reels that go viral on TikTok, Reels, and Shorts.

You receive:
- Metadata for each available raw clip/photo
- A rigorous analysis of each clip (subjects, action, scores, narrative role, speech, audio)

You decide:
- Which clips to use, in what order, with in/out timestamps
- How to handle each clip's ORIGINAL audio (keep_loud / keep_natural / duck / mute / subtitle_only)
- Text overlays on the timeline (hook, subtitles, captions, punchlines, closer)
- SFX cues (whooshes, impacts, swells)
- Music mood + genre + BPM target
- Whether voiceover is needed (only if no clip carries the story)
- Upload caption + hashtags


STORY DOCTRINE (how reels actually earn watches):
- PAYOFF FIRST. Find the single strongest moment in the footage before structuring anything. No strong payoff = shorter reel around the best real moment, never a faked one. Build the cut backwards from the payoff.
- HOOK IS A PROMISE. First 0.5s = most arresting frame + (optionally) a caption that opens a curiosity gap. A hook states tension, never a conclusion.
- TENSION LINE. Every clip between hook and payoff escalates toward it. A clip that does not build gets cut.
- AUDIO SPINE FOR MUSIC EVENTS. Live music must never be chopped. Pick the hero clip with the best continuous audio and keep one unbroken window of it under the whole reel; cut video over it. Sync the spine clip's own video segments to the spine timeline.
- CAPTION CRAFT. Max 2 captions: open a gap or heighten anticipation. Never decorate, conclude, or assert unverifiable facts. The payoff gets no caption.

CORE PRINCIPLES:

0. **best_moments are SACRED.** When a clip's analysis includes best_moments=[(s,e), ...], your in_s/out_s for that clip MUST be inside one of those windows. They are the only times the analyst (you, on a previous pass) judged the footage worth using. Picking arbitrary segments is the #1 cause of weak reels. If none of a clip's best_moments fit the slot you need, pick a different clip.

1. **Audio first.** Most reels fail because they paper everything in music. If a clip has meaningful speech (clip.speech_is_meaningful=true AND clip.speech_quality >= 6), KEEP IT LOUD and duck music under it. The viewer should hear the human voice. Avoid cutting through a vocal mid-phrase.

2. Don't lie with subtitles. Use audio_decision='subtitle_only' ONLY when speech is meaningful but quality is poor (5-7) — mute the audio and surface the words as on-screen text.

3. **Hook in 0.5 seconds.** The first clip must be the single most arresting frame in the source set. No long fades. No throat-clearing. Pick from the highest aesthetic_score clip with non-empty best_moments — and use a best_moment timestamp.

4. **One emotional climax.** Build, peak, exit. Three acts of escalation. Don't flatline.

5. Cut on motion, hit on impact. Match cuts to physical movement when possible. Place SFX 'impact' on top of beat drops if music is rhythmic. Don't pile SFX on top of meaningful speech.

6. Vertical math: 1080x1920. Crop_hint per clip ('face_left', 'face_center', 'face_right', 'center', 'top', 'bottom') guides the smart-crop stage.

7. Length: 20-35 seconds. Hard cap 40. If you can tell it in 20, do.

8. **Text restraint.** At most 3 overlays total. The hook is non-negotiable; everything else must earn its place. Overlays appear OFF the subject's face — they don't cover what people want to see.

9. Music mood follows the footage. Don't impose hype on quiet intimacy. Don't impose lo-fi on a rave.

10. Voiceover is a last resort. Only if zero usable diegetic speech AND the story is hard to read visually.

11. **Close clip floor.** For role='close', prefer a clip with aesthetic_score >= 5. If no clip qualifies, end on a video clip's natural fade rather than a weak still photo. NEVER use a blurry/dark photo for close — it sinks the entire reel's perceived quality.

12. **Hook clip floor.** For role='hook', aesthetic_score >= 4 minimum. The hook must have a recognizable subject — face, action, environment.

13. **Source quality realism.** If ALL available footage has aesthetic_score <= 4, you are working with B-grade material. Make a tight 15-20 second cut emphasizing the best 2-3 moments. Don't pad to 30s with weak filler.

14. **NO EM DASHES.** Never use em dash (—) in hook_text, overlay text, captions, voiceover_script, or upload_caption. Use commas, ellipses, or new sentences instead. This is a hard rule. Hyphens are fine in hyphenated words; en dashes are not allowed either.

15. **Don't invent facts.** Overlay text and captions must not assert details you can't verify from the clips or metadata: no seasons, city names, event names, or people's names unless they appear in the clip analyses, speech, or shot_at timestamps. "summer nights" on an October event kills trust. Evocative but factless beats specific but wrong.

16. **Overlays must fit the timeline.** Every overlay's start_s + duration_s must be within the total reel duration. A closer scheduled after the final frame never displays."""


_CASUAL_ADDENDUM = """

## MODE: CASUAL

The user shot a folder of phone clips with no specific direction. Your job is to discover the story from the footage itself.

- Auto-detect the event type from clip subjects/setting (concert, trip, wedding, party, casual, sports, family, food, etc.).
- Invent a hook that captures what makes this moment shareable.
- Default to the broadest emotional appeal of the footage — what would make a stranger pause scrolling?
- Lean on diegetic audio (real sounds from the event). The viewer wants to feel they were there.
- Keep text overlays sparse. The user has no taste signal — over-text feels generic.
- Output ONE strong cut, not three weak options."""


_PRO_ADDENDUM_TEMPLATE = """

## MODE: PRO

The user is a content creator with intentional footage and a brief. Your job is to EXECUTE their vision, not invent your own.

The user's brief is the source of truth. If they say "make it cinematic, focus on the build", you focus on the build moment with cinematic pacing, regardless of what the footage looks like at first glance.

### USER BRIEF
{brief}

{reference_section}

{brand_section}

PRO-MODE PRINCIPLES (override casual defaults if they conflict):
- Brief > footage charm. If the brief says "intense and dark", don't pick the goofy moment even if it's the highest-energy clip.
- Match the pacing and cut frequency the creator typically uses (from references when provided).
- Use the brand's text style preset for overlays.
- If brand kit specifies a voice profile, prefer voiceover_script when narration helps the brief land.
- Maintain consistency with the creator's series, not generic short-form best practices.

A-ROLL / B-ROLL DISCIPLINE:
- Each clip has a shot_type. Treat 'a_roll' as the spine of the cut — the talking head carries the narrative.
- Cut to 'b_roll', 'performance', 'crowd', 'establishing' during A-roll's natural pauses for illustration, never mid-phrase.
- When a script is provided below, follow it. Each script BEAT has a matched A-roll window (or is unmatched, requiring B-roll). Build the arc as: BEAT 0 A-roll → B-roll illustration → BEAT 1 A-roll → ... and so on.
- When no A-roll exists in the source set (e.g. concert recordings, pure b-roll vacation), ignore script alignment and fall back to standard story building."""


_SCHEMA_HINT = """JSON schema (informal):

{
  "event_type": "concert" | "trip" | "wedding" | "party" | "casual" | "...",
  "target_duration_s": 28.0,
  "arc": [
    {
      "asset_path": "<path string from input>",
      "in_s": 0.0,
      "out_s": 1.4,
      "crop_hint": "face_center",
      "role": "hook" | "setup" | "energy" | "climax" | "transition" | "close",
      "audio_decision": "keep_loud" | "keep_natural" | "duck" | "mute" | "subtitle_only",
      "original_audio_gain_db": 0.0,
      "transition_in": "cut" | "fade",   // fade ONLY into the close or a big emotional shift. Default cut.
      "punch_in": false                   // true = slow zoom-in. Use on held/static shots and photos to add life.
    },
    ...
  ],
  "audio_spine": {"asset_path": "<path>", "in_s": 46.0, "out_s": 64.0} | null,   // REQUIRED for live-music footage. Spine length = reel length. Sync the spine clip's own video segments to spine time.
  "music_mood": "uplifting | melancholic | hype | dreamy | gritty | tender | ...",
  "music_genre": "indie-electronic | hyperpop | lofi | trap | orchestral | ...",
  "music_bpm": 128,
  "music_path": null,
  "music_gain_db": -10.0,
  "music_ducks_under_speech": true,
  "overlays": [
    {
      "type": "hook" | "subtitle" | "caption" | "punchline" | "lower_third" | "closer",
      "text": "string (kept tight)",
      "start_s": 0.0,
      "duration_s": 2.0,
      "style": "bold_center"
    }
  ],
  "sfx": [
    {"sfx": "whoosh" | "impact" | "pop" | "shimmer" | "click" | "boom" | "swell" | "riser",
     "at_s": 0.0, "volume_db": -6.0}
  ],
  "voiceover_script": null | "string",
  "voiceover_voice_id": null,
  "upload_caption": "string",
  "hashtags": ["#one", "#two"]
}
"""


def _format_clip_summary(analyses: list[ClipAnalysis], assets_by_path: dict[Path, MediaAsset]) -> str:
    """Compact, scannable representation for the LLM."""
    lines = []
    for a in analyses:
        asset = assets_by_path.get(a.asset_path)
        dur = f"{asset.duration_s:.1f}s" if asset and asset.type == "video" else "PHOTO"
        shot_at = (
            asset.timestamp.strftime("%Y-%m-%d %H:%M")
            if asset and asset.timestamp
            else "unknown"
        )
        speech = ""
        if a.has_speech:
            mean = "MEANINGFUL" if a.speech_is_meaningful else "filler"
            speech = f" speech={mean}/q{a.speech_quality:.0f}"
            if a.speech_segments:
                seg_texts = []
                for s in a.speech_segments[:3]:
                    seg_texts.append(f"[{s.start_s:.1f}-{s.end_s:.1f}] \"{s.text}\"")
                speech += " " + " ".join(seg_texts)
        moments = ", ".join(f"({s:.1f}-{e:.1f})" for s, e in a.best_moments[:3])
        lines.append(
            f"- path={a.asset_path} [{dur}] shot_at={shot_at} shot={a.shot_type} role={a.narrative_role} "
            f"comp={a.composition_score:.0f} aesth={a.aesthetic_score:.0f} energy={a.energy:.0f} "
            f"motion={a.motion_type} light={a.lighting} ambient={a.ambient_quality:.0f}{speech} "
            f"best={[moments]} | {a.action} | {a.emotion} | {a.setting}"
        )
        if a.audio_notes:
            lines.append(f"  audio_notes: {a.audio_notes}")
        if a.notes:
            lines.append(f"  notes: {a.notes}")
    return "\n".join(lines)


def _build_system_prompt(
    mode: str,
    brief: str | None,
    ref_style: RefStyle | None,
    brand_kit: BrandKit | None,
) -> str:
    if mode == "casual":
        return _SYSTEM_CORE + _CASUAL_ADDENDUM
    # PRO mode
    ref_section = ""
    if ref_style:
        ref_section = (
            "### REFERENCE STYLE (extracted from creator's reference reels)\n"
            f"- Summary: {ref_style.summary}\n"
            f"- Pacing: {ref_style.pacing}\n"
            f"- Energy: {ref_style.energy}\n"
            f"- Typical overlay style: {ref_style.typical_overlay_style}\n"
            f"- Approx cut frequency: {ref_style.cut_frequency_hz} cuts/sec\n"
        )
    brand_section = ""
    if brand_kit:
        brand_section = (
            "### BRAND KIT\n"
            f"- Name: {brand_kit.name}\n"
            f"- Watermark text: {brand_kit.watermark_text or 'none'}\n"
            f"- Voice profile available: {bool(brand_kit.voice_id)}\n"
            f"- Intro sting: {bool(brand_kit.intro_sting_path)}\n"
            f"- Outro sting: {bool(brand_kit.outro_sting_path)}\n"
        )
    return _SYSTEM_CORE + _PRO_ADDENDUM_TEMPLATE.format(
        brief=brief or "(none provided — fall back to casual instincts)",
        reference_section=ref_section,
        brand_section=brand_section,
    )


def build_edl(
    assets: list[MediaAsset],
    analyses: list[ClipAnalysis],
    target_duration_s: float = 28.0,
    model: str = DEFAULT_MODEL,
    user_brief: str | None = None,
    mode: str = "casual",
    ref_style: RefStyle | None = None,
    brand_kit: BrandKit | None = None,
    script: Script | None = None,
    variant_angle: str | None = None,
    _retry: bool = True,
) -> EDL:
    """Ask Claude Opus to produce an EDL given the analyzed clips."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY missing from environment")
    client = Anthropic(api_key=api_key)

    assets_by_path = {a.path: a for a in assets}
    clip_summary = _format_clip_summary(analyses, assets_by_path)

    user_msg_parts = [
        f"Target reel length: {target_duration_s:.0f} seconds (hard cap 40).",
        "",
        "Available clips:",
        clip_summary,
        "",
    ]
    if script is not None and script.beats:
        user_msg_parts += [
            "Script alignment (creator's narration, pre-matched to A-roll windows):",
            _format_script(script),
            "",
            "Build the EDL to follow the script: each BEAT becomes an A-roll segment "
            "using the matched window, interleaved with B-roll/performance/crowd shots "
            "between beats for visual variety. UNMATCHED beats need B-roll illustration "
            "OR a voiceover_script line. Total reel length should still fit the target.",
            "",
        ]
    if variant_angle:
        user_msg_parts.insert(0, f"Editorial angle for THIS take: {variant_angle}\n")
    user_msg_parts += [
        "Output JSON matching this schema (informal):",
        _SCHEMA_HINT,
    ]

    system_prompt = _build_system_prompt(mode, user_brief, ref_style, brand_kit)

    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system_prompt,
        messages=[{"role": "user", "content": "\n".join(user_msg_parts)}],
    )

    text = "".join(b.text for b in msg.content if hasattr(b, "text"))

    # Strip code fences if present
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    text = text.strip()

    try:
        data = json.loads(text)
        edl = EDL.model_validate(data)
    except (json.JSONDecodeError, ValidationError) as e:
        raise RuntimeError(
            f"Story stage produced invalid EDL: {e}\n\nRaw output:\n{text}"
        ) from e

    # ENFORCE the audio spine: doctrine says live music must never chop, and
    # the model skips it often enough that the rule needs teeth. One retry with
    # an explicit demand; judged output doubles in quality with a spine.
    has_live_music = any(a.has_music for a in analyses)
    if has_live_music and edl.audio_spine is None and _retry:
        followup = (
            "Your EDL omitted audio_spine but the footage contains live music. "
            "This is a hard requirement. Choose the clip whose analysis shows the "
            "best continuous audio (performer and crowd both clear), set audio_spine "
            "to one unbroken window of it (length = total reel length), time-sync "
            "that clip's own video segments to the spine window, and ensure any "
            "caption text is coherent with continuous audio. Output the corrected "
            "full EDL JSON only."
        )
        msg2 = client.messages.create(
            model=model,
            max_tokens=4096,
            system=system_prompt,
            messages=[
                {"role": "user", "content": "\n".join(user_msg_parts)},
                {"role": "assistant", "content": text},
                {"role": "user", "content": followup},
            ],
        )
        text2 = "".join(b.text for b in msg2.content if hasattr(b, "text")).strip()
        if text2.startswith("```"):
            text2 = text2.split("\n", 1)[1]
            if text2.rstrip().endswith("```"):
                text2 = text2.rstrip()[:-3]
        try:
            edl = EDL.model_validate(json.loads(text2.strip()))
        except (json.JSONDecodeError, ValidationError):
            pass  # keep the first EDL rather than fail the run
    return edl


def build_variants(
    assets: list[MediaAsset],
    analyses: list[ClipAnalysis],
    n: int,
    briefs: list[str] | None = None,
    mode: str = "pro",
    ref_style: RefStyle | None = None,
    brand_kit: BrandKit | None = None,
    script: Script | None = None,
    target_duration_s: float = 28.0,
    model: str = DEFAULT_MODEL,
) -> list[EDL]:
    """Generate N distinct EDLs by varying brief OR temperature.

    If `briefs` is provided, use one brief per variant. Otherwise vary the
    editorial angle per take (temperature is deprecated on current Opus).
    """
    _ANGLES = [
        "maximum energy, fastest cuts, lead with the single most explosive moment",
        "cinematic and restrained, longer holds, let the emotion breathe",
        "story-first, clear three-act arc, prioritize meaningful speech",
        "playful and meme-aware, lean into the funniest beats",
        "intimate POV, favor close shots and quiet details",
    ]
    edls: list[EDL] = []
    for i in range(n):
        this_brief = briefs[i] if briefs and i < len(briefs) else None
        angle = None if this_brief else _ANGLES[i % len(_ANGLES)]
        edl = build_edl(
            assets=assets,
            analyses=analyses,
            target_duration_s=target_duration_s,
            model=model,
            user_brief=this_brief,
            mode=mode,
            ref_style=ref_style,
            brand_kit=brand_kit,
            script=script,
            variant_angle=angle,
        )
        edls.append(edl)
    return edls


def save(edl: EDL, path: Path, note: str = "") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(edl.model_dump_json(indent=2))
    # Version history with provenance: git-for-edits. v1..N never overwritten.
    vdir = path.parent / "versions"
    vdir.mkdir(exist_ok=True)
    n = len(list(vdir.glob("edl_v*.json"))) + 1
    payload = json.loads(edl.model_dump_json())
    payload["_provenance"] = {"version": n, "note": note}
    (vdir / f"edl_v{n}.json").write_text(json.dumps(payload, indent=2))


def compose_reel_timeline(edl: EDL, analyses: list) -> list[dict]:
    """The rendered reel's own timeline: reel_t -> source clip/time + events.
    Makes the CREATED media as described as the supplied media, so prompts
    like 'at 0:14 the shot is shaky' resolve mechanically."""
    by_path = {str(a.asset_path): a for a in analyses}
    out, t = [], 0.0
    for c in edl.arc:
        a = by_path.get(str(c.asset_path))
        evs = [
            {"reel_t": round(t + (e.t - c.in_s), 2), "v": e.v, "s": e.s}
            for e in (a.timeline if a else [])
            if c.in_s <= e.t <= c.out_s
        ]
        out.append({
            "reel_start": round(t, 2), "reel_end": round(t + c.duration_s, 2),
            "source": str(c.asset_path), "src_in": c.in_s, "src_out": c.out_s,
            "role": c.role.value if hasattr(c.role, "value") else str(c.role),
            "audio": str(c.audio_decision.value if hasattr(c.audio_decision, "value") else c.audio_decision),
            "events": evs,
        })
        t += c.duration_s
    return out


def load(path: Path) -> EDL:
    return EDL.model_validate_json(path.read_text())
