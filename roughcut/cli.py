"""montage CLI.

Stage-by-stage for fast iteration; or one-shot `process`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from roughcut import ingest as ingest_mod
from roughcut import understand as understand_mod
from roughcut import story as story_mod
from roughcut import music as music_mod
from roughcut import speech as speech_mod
from roughcut import sfx_gen
from roughcut import edit as edit_mod
from roughcut import brand as brand_mod
from roughcut import references as ref_mod
from roughcut import script as script_mod
from roughcut.assemble import assemble
from roughcut.schemas import EDL, MediaAsset, ClipAnalysis, SpeechSegment, BrandKit, RefStyle

load_dotenv()

app = typer.Typer(no_args_is_help=True, help="Camera roll → finished Reel")
console = Console()

CACHE_DIR = Path("cache")
ASSETS_JSON = CACHE_DIR / "assets.json"
SPEECH_JSON = CACHE_DIR / "speech.json"
ANALYSES_JSON = CACHE_DIR / "analyses.json"
EDL_JSON = CACHE_DIR / "edl.json"
REFSTYLE_JSON = CACHE_DIR / "refstyle.json"


def _save_speech(speech_by_path: dict[Path, list[SpeechSegment]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        str(p): [s.model_dump() for s in segs]
        for p, segs in speech_by_path.items()
    }
    path.write_text(json.dumps(payload, indent=2))


def _load_speech(path: Path) -> dict[Path, list[SpeechSegment]]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    return {
        Path(k): [SpeechSegment.model_validate(d) for d in v]
        for k, v in raw.items()
    }


def _save_assets(assets: list[MediaAsset], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([a.model_dump(mode="json") for a in assets], indent=2, default=str))


def _load_assets(path: Path) -> list[MediaAsset]:
    return [MediaAsset.model_validate(d) for d in json.loads(path.read_text())]


@app.command()
def ingest(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    out: Path = typer.Option(ASSETS_JSON, "--out"),
) -> None:
    """Walk a media folder, probe each file, cache metadata."""
    assets = ingest_mod.ingest_folder(input_dir)
    _save_assets(assets, out)

    t = Table(title=f"{len(assets)} assets ingested")
    t.add_column("file"); t.add_column("type"); t.add_column("dim")
    t.add_column("dur"); t.add_column("audio")
    for a in assets:
        t.add_row(
            a.path.name, a.type, f"{a.width}x{a.height}",
            f"{a.duration_s:.1f}s" if a.type == "video" else "—",
            "yes" if a.has_audio else "no",
        )
    console.print(t)
    console.print(f"[green]Saved[/green] → {out}")


@app.command()
def speech(
    assets_in: Path = typer.Option(ASSETS_JSON, "--assets"),
    out: Path = typer.Option(SPEECH_JSON, "--out"),
    work_dir: Path = typer.Option(Path("cache/audio"), "--work"),
) -> None:
    """Whisper transcribes every video's audio. Ground-truth speech segments."""
    assets = _load_assets(assets_in)
    videos = [a for a in assets if a.type == "video" and a.has_audio]
    if not videos:
        console.print("[yellow]No videos with audio. Skipping speech.[/yellow]")
        _save_speech({}, out)
        return
    speech_by_path = speech_mod.transcribe_all(videos, work_dir=work_dir)
    _save_speech(speech_by_path, out)
    total_segs = sum(len(v) for v in speech_by_path.values())
    console.print(f"[green]Transcribed[/green] {len(speech_by_path)} videos, {total_segs} segments → {out}")


@app.command()
def understand(
    assets_in: Path = typer.Option(ASSETS_JSON, "--assets"),
    speech_in: Path = typer.Option(SPEECH_JSON, "--speech"),
    out: Path = typer.Option(ANALYSES_JSON, "--out"),
    concurrency: int = typer.Option(4, "--concurrency"),
    model: str = typer.Option(
        os.getenv("GEMINI_MODEL", "gemini-2.5-pro"), "--model"
    ),
) -> None:
    """Gemini analyzes every cached asset (uses Whisper transcripts if present)."""
    assets = _load_assets(assets_in)
    speech_by_path = _load_speech(speech_in)
    analyses = understand_mod.analyze_all_sync(
        assets, model=model, concurrency=concurrency, speech_by_path=speech_by_path
    )
    understand_mod.save(analyses, out)
    console.print(f"[green]Analyzed[/green] {len(analyses)}/{len(assets)} → {out}")


@app.command("init-sfx")
def init_sfx() -> None:
    """Generate the synthesized SFX library WAVs."""
    paths = sfx_gen.generate_all()
    for p in paths:
        console.print(f"  {p}")
    console.print(f"[green]Generated[/green] {len(paths)} SFX files")


@app.command()
def edit(
    instruction: str = typer.Argument(..., help="Natural-language edit instruction"),
    edl_in: Path = typer.Option(EDL_JSON, "--edl"),
    analyses_in: Path = typer.Option(ANALYSES_JSON, "--analyses"),
    out_edl: Path = typer.Option(EDL_JSON, "--out-edl"),
    render_now: bool = typer.Option(True, "--render/--no-render"),
    out_video: Path = typer.Option(Path("output/reel.mp4"), "--out"),
    music: Path = typer.Option(None, "--music"),
) -> None:
    """Apply a natural-language edit instruction to the current EDL.

    Example: montage edit "at 0:08 add text 'WAIT FOR IT' for 2 seconds"
    Example: montage edit "remove the second SFX and trim the close by 1 second"
    """
    edl = story_mod.load(edl_in)
    analyses = understand_mod.load(analyses_in)
    console.rule("[bold]PARSE")
    ops = edit_mod.parse_edits(instruction, edl, analyses)
    for op in ops:
        console.print(f"  • {edit_mod.describe_op(op)}")
    if not ops:
        console.print("[yellow]No ops produced[/yellow]")
        return

    console.rule("[bold]APPLY")
    new_edl = edit_mod.apply_edits(edl, ops)

    # Preserve music_path if user didn't override
    if music is not None:
        new_edl = new_edl.model_copy(update={"music_path": music})
    elif edl.music_path is not None and new_edl.music_path is None:
        new_edl = new_edl.model_copy(update={"music_path": edl.music_path})

    story_mod.save(new_edl, out_edl)
    _print_edl(new_edl)
    console.print(f"[green]EDL[/green] → {out_edl}")

    if render_now:
        console.rule("[bold]RENDER")
        assets = _load_assets(ASSETS_JSON)
        if new_edl.music_path is not None:
            track = music_mod.analyze_track(new_edl.music_path)
            new_edl = music_mod.snap_edl_to_beats(new_edl, track, snap_strength=0.7)
        voiceover = None
        if new_edl.voiceover_script:
            vo_path = Path("cache/work/voiceover.wav")
            voiceover = music_mod.generate_voiceover(
                new_edl.voiceover_script, vo_path, voice_id=new_edl.voiceover_voice_id
            )
        final = assemble(new_edl, assets, out_path=out_video, voiceover_path=voiceover)
        console.print(f"[green]Done[/green] → {final}")


@app.command()
def story(
    assets_in: Path = typer.Option(ASSETS_JSON, "--assets"),
    analyses_in: Path = typer.Option(ANALYSES_JSON, "--analyses"),
    out: Path = typer.Option(EDL_JSON, "--out"),
    duration: float = typer.Option(28.0, "--duration"),
    brief: str = typer.Option("", "--brief", help="Optional user brief"),
    model: str = typer.Option(
        os.getenv("ANTHROPIC_MODEL", "claude-opus-4-7"), "--model"
    ),
) -> None:
    """Claude Opus builds an Edit Decision List."""
    assets = _load_assets(assets_in)
    analyses = understand_mod.load(analyses_in)
    edl = story_mod.build_edl(
        assets, analyses,
        target_duration_s=duration,
        user_brief=brief or None,
        model=model,
    )
    story_mod.save(edl, out)
    _print_edl(edl)
    console.print(f"[green]EDL[/green] → {out}")


@app.command()
def render(
    edl_in: Path = typer.Option(EDL_JSON, "--edl"),
    assets_in: Path = typer.Option(ASSETS_JSON, "--assets"),
    music: Path = typer.Option(None, "--music", help="Music track WAV/MP3"),
    out: Path = typer.Option(Path("output/reel.mp4"), "--out"),
    work_dir: Path = typer.Option(Path("cache/work"), "--work"),
    snap: float = typer.Option(0.7, "--snap", help="Beat-snap strength 0-1"),
    watermark: str = typer.Option("", "--watermark", help="Watermark text (e.g. 'made with roughcut')"),
    brand_path: Path = typer.Option(None, "--brand"),
) -> None:
    """Render the final MP4 from cached EDL."""
    assets = _load_assets(assets_in)
    edl = story_mod.load(edl_in)

    brand = brand_mod.load_brand(brand_path) if brand_path else None

    if music is not None:
        track = music_mod.analyze_track(music)
        console.print(f"Track: {track.bpm:.1f} BPM, {len(track.beat_times)} beats")
        edl = music_mod.snap_edl_to_beats(edl, track, snap_strength=snap)
        edl = edl.model_copy(update={"music_path": music})

    voiceover = None
    if edl.voiceover_script:
        vo_path = work_dir / "voiceover.wav"
        voiceover = music_mod.generate_voiceover(
            edl.voiceover_script, vo_path, voice_id=edl.voiceover_voice_id
        )
        if voiceover:
            console.print(f"Voiceover → {voiceover}")
        else:
            console.print("[yellow]Voiceover skipped (ELEVENLABS_API_KEY missing)[/yellow]")

    final = assemble(
        edl, assets, out_path=out, work_dir=work_dir,
        voiceover_path=voiceover, brand=brand,
        watermark_text=watermark or None,
    )
    console.print(f"[green]Done[/green] → {final}")


def _run_ingest_speech_understand(input_dir: Path, concurrency: int):
    """Stages 1-3 shared by casual and pro."""
    console.rule("[bold]1/5 INGEST")
    assets = ingest_mod.ingest_folder(input_dir)
    _save_assets(assets, ASSETS_JSON)
    console.print(f"{len(assets)} assets")

    console.rule("[bold]2/5 SPEECH (Whisper)")
    videos = [a for a in assets if a.type == "video" and a.has_audio]
    speech_by_path = speech_mod.transcribe_all(videos, work_dir=Path("cache/audio"))
    _save_speech(speech_by_path, SPEECH_JSON)
    total_segs = sum(len(v) for v in speech_by_path.values())
    console.print(f"{len(speech_by_path)} videos transcribed, {total_segs} segments")

    console.rule("[bold]3/5 UNDERSTAND")
    analyses = understand_mod.analyze_all_sync(
        assets, concurrency=concurrency, speech_by_path=speech_by_path
    )
    understand_mod.save(analyses, ANALYSES_JSON)
    console.print(f"{len(analyses)} analyses")
    return assets, analyses


def _render(
    edl: EDL,
    assets: list[MediaAsset],
    music: Path | None,
    out: Path,
    brand: BrandKit | None = None,
    watermark_text: str | None = None,
) -> None:
    if music is not None:
        track = music_mod.analyze_track(music)
        console.print(f"Track: {track.bpm:.1f} BPM, {len(track.beat_times)} beats")
        edl = music_mod.snap_edl_to_beats(edl, track, snap_strength=0.7)
        edl = edl.model_copy(update={"music_path": music})
    voiceover = None
    if edl.voiceover_script:
        vo_path = Path("cache/work/voiceover.wav")
        voice_id = edl.voiceover_voice_id or (brand.voice_id if brand else None)
        voiceover = music_mod.generate_voiceover(
            edl.voiceover_script, vo_path, voice_id=voice_id
        )
    final = assemble(
        edl, assets, out_path=out, voiceover_path=voiceover,
        brand=brand, watermark_text=watermark_text,
    )
    console.print(f"[bold green]Done[/bold green] → {final}")


@app.command()
def casual(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    music: Path = typer.Option(None, "--music", help="Optional music track"),
    out: Path = typer.Option(Path("output/reel.mp4"), "--out"),
    duration: float = typer.Option(25.0, "--duration"),
    concurrency: int = typer.Option(4, "--concurrency"),
    no_watermark: bool = typer.Option(False, "--no-watermark", help="Paid tier removes watermark"),
) -> None:
    """ICP 1: Amateur. Drop a folder, get a finished Reel. Zero config.

    Free tier ships with a corner watermark. --no-watermark for paid users.
    """
    assets, analyses = _run_ingest_speech_understand(input_dir, concurrency)

    console.rule("[bold]4/5 STORY (casual)")
    edl = story_mod.build_edl(
        assets, analyses,
        target_duration_s=duration,
        mode="casual",
    )
    story_mod.save(edl, EDL_JSON)
    _print_edl(edl)

    console.rule("[bold]5/5 RENDER")
    watermark = None if no_watermark else "made with roughcut"
    _render(edl, assets, music, out, watermark_text=watermark)


@app.command()
def pro(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    brief: str = typer.Option(..., "--brief", help="Required: direction for the cut"),
    music: Path = typer.Option(None, "--music"),
    out: Path = typer.Option(Path("output/reel.mp4"), "--out"),
    duration: float = typer.Option(28.0, "--duration"),
    refs: list[Path] = typer.Option([], "--ref", help="Reference reels to mimic (repeatable)"),
    brand_path: Path = typer.Option(None, "--brand", help="BrandKit JSON path"),
    script_path: Path = typer.Option(
        None, "--script", help="Script text file — A-roll auto-aligned"
    ),
    variants: int = typer.Option(1, "--variants", min=1, max=5),
    concurrency: int = typer.Option(4, "--concurrency"),
) -> None:
    """ICP 2: Creator. Brief-driven. References, brand kit, script alignment, multi-variant."""
    assets, analyses = _run_ingest_speech_understand(input_dir, concurrency)

    ref_style = None
    if refs:
        console.rule(f"[bold]REFS ({len(refs)})")
        ref_style = ref_mod.analyze(refs)
        ref_mod.save(ref_style, REFSTYLE_JSON)
        console.print(f"  Style: {ref_style.summary[:120]}…")

    brand = None
    if brand_path is not None:
        brand = brand_mod.load_brand(brand_path)
        console.print(f"  Brand: [cyan]{brand.name}[/cyan]")

    script = None
    if script_path is not None:
        console.rule("[bold]SCRIPT")
        raw = script_mod.load_script(script_path)
        speech_by_path = _load_speech(SPEECH_JSON)
        script = script_mod.align_script(raw, speech_by_path, analyses)
        matched = sum(1 for b in script.beats if b.a_roll_asset_path is not None)
        a_roll_count = sum(1 for a in analyses if a.shot_type == "a_roll")
        console.print(
            f"  {len(script.beats)} beats, {matched} matched to A-roll, "
            f"{a_roll_count} A-roll clips available"
        )
        script_mod.save(script, CACHE_DIR / "script.json")

    console.rule(f"[bold]4/5 STORY (pro, variants={variants})")
    if variants > 1:
        edls = story_mod.build_variants(
            assets, analyses, n=variants,
            mode="pro", ref_style=ref_style, brand_kit=brand, script=script,
            target_duration_s=duration,
        )
        for i, edl in enumerate(edls):
            story_mod.save(edl, CACHE_DIR / f"edl_v{i+1}.json")
            console.print(f"\n[bold]Variant {i+1}[/bold]")
            _print_edl(edl)
        console.rule("[bold]5/5 RENDER ALL")
        out_dir = out.parent
        for i, edl in enumerate(edls):
            variant_out = out_dir / f"{out.stem}_v{i+1}{out.suffix}"
            _render(edl, assets, music, variant_out, brand=brand)
    else:
        edl = story_mod.build_edl(
            assets, analyses,
            target_duration_s=duration,
            user_brief=brief,
            mode="pro",
            ref_style=ref_style,
            brand_kit=brand,
            script=script,
        )
        story_mod.save(edl, EDL_JSON)
        _print_edl(edl)
        console.rule("[bold]5/5 RENDER")
        _render(edl, assets, music, out, brand=brand)


@app.command("brand-init")
def brand_init(
    out: Path = typer.Option(Path("brand.json"), "--out"),
) -> None:
    """Create a starter BrandKit JSON. Edit it to set logo, fonts, watermark."""
    b = brand_mod.init_default(out)
    console.print(f"[green]Wrote[/green] {out}")
    console.print("Edit it to set: logo_path, font_primary_path, voice_id, watermark_text, etc.")


@app.command()
def process(
    input_dir: Path = typer.Argument(..., exists=True, file_okay=False),
    music: Path = typer.Option(None, "--music"),
    out: Path = typer.Option(Path("output/reel.mp4"), "--out"),
    duration: float = typer.Option(28.0, "--duration"),
    brief: str = typer.Option("", "--brief"),
    concurrency: int = typer.Option(4, "--concurrency"),
) -> None:
    """[DEPRECATED — use `casual` or `pro`] End-to-end: ingest → understand → story → render."""
    console.rule("[bold]1/5 INGEST")
    assets = ingest_mod.ingest_folder(input_dir)
    _save_assets(assets, ASSETS_JSON)
    console.print(f"{len(assets)} assets")

    console.rule("[bold]2/5 SPEECH (Whisper)")
    videos = [a for a in assets if a.type == "video" and a.has_audio]
    speech_by_path = speech_mod.transcribe_all(videos, work_dir=Path("cache/audio"))
    _save_speech(speech_by_path, SPEECH_JSON)
    total_segs = sum(len(v) for v in speech_by_path.values())
    console.print(f"{len(speech_by_path)} videos transcribed, {total_segs} segments")

    console.rule("[bold]3/5 UNDERSTAND")
    analyses = understand_mod.analyze_all_sync(
        assets, concurrency=concurrency, speech_by_path=speech_by_path
    )
    understand_mod.save(analyses, ANALYSES_JSON)
    console.print(f"{len(analyses)} analyses")

    console.rule("[bold]4/5 STORY")
    edl = story_mod.build_edl(
        assets, analyses,
        target_duration_s=duration,
        user_brief=brief or None,
    )
    story_mod.save(edl, EDL_JSON)
    _print_edl(edl)

    console.rule("[bold]5/5 RENDER")
    if music is not None:
        track = music_mod.analyze_track(music)
        console.print(f"Track: {track.bpm:.1f} BPM, {len(track.beat_times)} beats")
        edl = music_mod.snap_edl_to_beats(edl, track, snap_strength=0.7)
        edl = edl.model_copy(update={"music_path": music})

    voiceover = None
    if edl.voiceover_script:
        vo_path = Path("cache/work/voiceover.wav")
        voiceover = music_mod.generate_voiceover(
            edl.voiceover_script, vo_path, voice_id=edl.voiceover_voice_id
        )

    final = assemble(edl, assets, out_path=out, voiceover_path=voiceover)
    console.print(f"[bold green]Done[/bold green] → {final}")


def _print_edl(edl: EDL) -> None:
    console.print(f"Event: [cyan]{edl.event_type}[/cyan]  target={edl.target_duration_s:.1f}s  "
                  f"music={edl.music_genre}/{edl.music_mood}@{edl.music_bpm}bpm")
    t = Table(title=f"EDL ({len(edl.arc)} clips, {edl.total_duration_s:.1f}s)")
    t.add_column("#"); t.add_column("clip"); t.add_column("in→out"); t.add_column("role")
    t.add_column("audio"); t.add_column("crop")
    for i, c in enumerate(edl.arc):
        t.add_row(
            str(i),
            c.asset_path.name,
            f"{c.in_s:.2f}→{c.out_s:.2f} ({c.duration_s:.2f}s)",
            c.role,
            c.audio_decision,
            c.crop_hint,
        )
    console.print(t)
    if edl.overlays:
        console.print(f"Overlays: {len(edl.overlays)}  | first: {edl.overlays[0].text!r}")
    if edl.voiceover_script:
        console.print(f"[yellow]Voiceover[/yellow]: {edl.voiceover_script[:80]}…")


if __name__ == "__main__":
    app()
