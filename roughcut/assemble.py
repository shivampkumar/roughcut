"""Assembly stage. Takes EDL + CropPlans + music + (optional) voiceover, runs ffmpeg.

Strategy: render each EDL clip to a normalized 1080x1920 30fps intermediate,
concat them, then mix the audio track in a final pass. Text overlays burn in
during the final pass using drawtext filters from each TextOverlay.

Multi-pass for debuggability. Optimize to single filter_complex later if needed.
"""

from __future__ import annotations

import shlex
import subprocess
import tempfile
from pathlib import Path

from roughcut.crop import CropPlan, plan_crops
from roughcut.music import sfx_path
from roughcut.schemas import BrandKit, EDL, EDLClip, MediaAsset, SFXCue, TextOverlay
from roughcut.text_render import render_all_overlays

TARGET_W_OUT = 1080
TARGET_H_OUT = 1920

TARGET_W = 1080
TARGET_H = 1920
FPS = 30

# Audio gains per decision, in dB applied to the clip's original track.
_AUDIO_GAIN = {
    "keep_loud": 0.0,
    "keep_natural": -3.0,
    "duck": -15.0,
    "mute": -100.0,        # effectively silence
    "subtitle_only": -100.0,
}


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    r = subprocess.run(args, capture_output=True, text=True, **kwargs)
    if r.returncode != 0:
        cmd_str = " ".join(repr(a) if " " in a or '"' in a else a for a in args)
        raise RuntimeError(
            f"ffmpeg failed (exit {r.returncode})\n\n"
            f"CMD:\n{cmd_str}\n\n"
            f"STDERR:\n{r.stderr}"
        )
    return r


def _render_clip(
    clip: EDLClip, asset: MediaAsset, crop: CropPlan, out_path: Path
) -> None:
    """Render one EDL clip as a normalized 1080x1920 30fps mp4."""
    gain_db = _AUDIO_GAIN.get(clip.audio_decision, -100.0) + clip.original_audio_gain_db
    has_audio = asset.has_audio and gain_db > -90

    if asset.type == "photo":
        # Photo: Ken Burns slow zoom + silent audio track (concat needs uniform streams).
        n_frames = max(2, int(round(clip.duration_s * FPS)))
        # Crop to 9:16 region, upscale for zoom headroom, then animate zoom 1.00 → 1.10.
        vf = (
            f"crop={crop.crop_w}:{crop.crop_h}:{crop.crop_x}:{crop.crop_y},"
            f"scale=1620:2880:flags=lanczos,"
            f"zoompan=z='min(1+0.10*on/{n_frames},1.10)'"
            f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={n_frames}:s={TARGET_W}x{TARGET_H}:fps={FPS},"
            f"setsar=1,format=yuv420p"
        )
        args = [
            "ffmpeg", "-y",
            "-i", str(asset.path),
            "-f", "lavfi", "-t", f"{clip.duration_s}",
            "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-vf", vf,
            "-frames:v", str(n_frames),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k",
            "-shortest",
            str(out_path),
        ]
        _run(args)
        return

    # Video: trim + crop + scale + audio handling.
    # ffmpeg auto-applies display rotation on decode in 5.x+, so crop coords
    # operate in display orientation (matching MediaAsset.width/height which we
    # already swap when rotation is ±90).
    chain = [crop.ffmpeg_filter()]
    if clip.punch_in:
        # Gentle continuous zoom-in: ~8% over the clip. pzoom carries state per frame.
        chain.append(
            "zoompan=z='min(pzoom+0.0006,1.08)':d=1"
            ":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":s={TARGET_W}x{TARGET_H}:fps={FPS}"
        )
    # Mild exposure/contrast consistency pass — phone clips from the same event
    # often have wildly different exposure; this narrows the gap without crushing.
    chain.append("normalize=strength=0.3:smoothing=50")
    chain.append(f"fps={FPS}")
    chain.append("format=yuv420p")
    vf = ",".join(chain)
    args = [
        "ffmpeg", "-y",
        "-ss", f"{clip.in_s}",
        "-i", str(asset.path),
        "-t", f"{clip.duration_s}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-r", str(FPS),
    ]
    if has_audio:
        # Linear gain factor from dB; add tiny fade in/out to prevent clip-boundary pops
        linear = 10 ** (gain_db / 20)
        fade_d = 0.05  # 50 ms — inaudible content-wise, kills pops
        fade_out_st = max(0.0, clip.duration_s - fade_d)
        args += [
            "-af",
            f"volume={linear:.4f},"
            f"afade=t=in:st=0:d={fade_d},"
            f"afade=t=out:st={fade_out_st:.3f}:d={fade_d},"
            f"aresample=async=1,asetpts=PTS-STARTPTS",
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        ]
    else:
        # Emit silent stereo track so concat doesn't choke
        args += [
            "-f", "lavfi", "-t", f"{clip.duration_s}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-shortest",
            "-c:a", "aac", "-b:a", "192k",
        ]
        # The -f lavfi input must come after primary -i. Restructure:
        args = [
            "ffmpeg", "-y",
            "-ss", f"{clip.in_s}",
            "-i", str(asset.path),
            "-f", "lavfi", "-t", f"{clip.duration_s}", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-t", f"{clip.duration_s}",
            "-vf", vf,
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
            "-r", str(FPS),
            "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        ]
    args.append(str(out_path))
    _run(args)


FADE_DUR = 0.25


def _concat_clips_xfade(
    clip_paths: list[Path],
    durations: list[float],
    transitions: list[str],
    out_path: Path,
) -> None:
    """Concat with per-boundary transitions. transitions[i] = transition INTO clip i.

    Crossfades shorten the timeline by FADE_DUR per fade — overlay/SFX timestamps
    drift slightly; acceptable at 0.25s when fades are rare (close only).
    """
    inputs: list[str] = []
    for p in clip_paths:
        inputs += ["-i", str(p)]

    n = len(clip_paths)
    parts: list[str] = []
    # Normalize every input to a common timebase/fps first — xfade refuses
    # mismatched timebases (concat filter outputs 1/1000000, raw streams 1/15360).
    for i in range(n):
        # fps first (it rewrites the timebase to 1/fps), then settb to unify.
        parts.append(f"[{i}:v]fps={FPS},settb=AVTB[v{i}n]")
        parts.append(f"[{i}:a]aresample=44100,asetpts=N/SR/TB[a{i}n]")

    vprev, aprev = "[v0n]", "[a0n]"
    timeline_end = durations[0]
    for i in range(1, n):
        vlab = f"[vx{i}]"
        alab = f"[ax{i}]"
        if transitions[i] == "fade":
            offset = max(0.0, timeline_end - FADE_DUR)
            parts.append(
                f"{vprev}[v{i}n]xfade=transition=fade:duration={FADE_DUR}:offset={offset:.3f}{vlab}"
            )
            parts.append(f"{aprev}[a{i}n]acrossfade=d={FADE_DUR}{alab}")
            timeline_end = offset + durations[i]
        else:
            parts.append(f"{vprev}[v{i}n]concat=n=2:v=1:a=0{vlab}")
            parts.append(f"{aprev}[a{i}n]concat=n=2:v=0:a=1{alab}")
            timeline_end += durations[i]
        vprev, aprev = vlab, alab

    args = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", ";".join(parts),
        "-map", vprev, "-map", aprev,
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        str(out_path),
    ]
    _run(args)


def _concat_clips(clip_paths: list[Path], out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        list_path = Path(f.name)
        for p in clip_paths:
            abs_path = p.resolve()
            # ffmpeg concat list: paths are relative to list file unless absolute.
            f.write(f"file {shlex.quote(str(abs_path))}\n")
    try:
        _run([
            "ffmpeg", "-y",
            "-f", "concat", "-safe", "0",
            "-i", str(list_path),
            "-c", "copy",
            str(out_path),
        ])
    finally:
        list_path.unlink(missing_ok=True)




def _brand_overlay_args(brand: BrandKit | None) -> tuple[list[str], str]:
    """Return (extra_inputs, filter_snippet_consuming_prev_label).

    Caller threads it after the text overlay chain.
    """
    if brand is None or brand.logo_path is None or not Path(brand.logo_path).exists():
        return [], ""
    return ["-i", str(brand.logo_path)], "BRAND_PLACEHOLDER"


def _mix_audio_and_overlays(
    video_in: Path,
    music_path: Path | None,
    voiceover_path: Path | None,
    overlays: list[TextOverlay],
    sfx_cues: list[SFXCue],
    music_gain_db: float,
    out_path: Path,
    brand: BrandKit | None = None,
    normalize_audio: bool = True,
    spine_audio: Path | None = None,
    total_duration_s: float = 0.0,
) -> None:
    """Single final pass: mix music + voiceover + SFX + clip audio, burn text overlays.
    With a spine, the spine IS the soundtrack: original clip audio and music
    are dropped entirely."""
    inputs: list[str] = ["-i", str(video_in)]  # 0: clips
    filter_parts: list[str] = []
    audio_labels: list[str] = ["[0:a]"]
    input_idx = 1

    if spine_audio is not None:
        inputs += ["-i", str(spine_audio)]
        filter_parts.append(f"[{input_idx}:a]aresample=44100[spine]")
        audio_labels = ["[spine]"]
        input_idx += 1
        music_path = None
        # Match the audio resolve with a 0.6s video fade to black at the end.
        spine_fade_vf = True

    if music_path is not None:
        inputs += ["-i", str(music_path)]
        gain = 10 ** (music_gain_db / 20)
        filter_parts.append(
            f"[{input_idx}:a]volume={gain:.4f},aresample=async=1,apad[mus]"
        )
        audio_labels.append("[mus]")
        input_idx += 1

    if voiceover_path is not None:
        inputs += ["-i", str(voiceover_path)]
        filter_parts.append(f"[{input_idx}:a]aresample=async=1,apad[vo]")
        audio_labels.append("[vo]")
        input_idx += 1

    # SFX cues — each becomes its own input + adelay (no apad, they're short)
    sfx_label_count = 0
    for cue in sfx_cues:
        path = sfx_path(cue.sfx)
        if path is None or not path.exists():
            continue
        inputs += ["-i", str(path)]
        gain = 10 ** (cue.volume_db / 20)
        delay_ms = int(cue.at_s * 1000)
        label = f"[sfx{sfx_label_count}]"
        filter_parts.append(
            f"[{input_idx}:a]volume={gain:.4f},"
            f"adelay={delay_ms}|{delay_ms}{label}"
        )
        audio_labels.append(label)
        input_idx += 1
        sfx_label_count += 1

    # amix everything → [amix]; then optional loudnorm → [aout]
    if len(audio_labels) > 1:
        filter_parts.append(
            "".join(audio_labels)
            + f"amix=inputs={len(audio_labels)}:dropout_transition=0:normalize=0[amix]"
        )
    else:
        filter_parts.append(f"{audio_labels[0]}anull[amix]")

    if normalize_audio:
        # Target -14 LUFS / -1 dBTP — TikTok/Reels/Shorts standard.
        filter_parts.append(
            "[amix]loudnorm=I=-14:TP=-1:LRA=11:linear=true:print_format=summary[aout]"
        )
    else:
        filter_parts.append("[amix]anull[aout]")

    # Video: render each overlay to PNG via PIL, composite via overlay filter,
    # then optionally composite the brand logo on top.
    prev_label = "[0:v]"
    text_overlay_count = 0

    if overlays:
        overlay_dir = out_path.parent / ".overlays"
        png_specs = render_all_overlays(overlays, overlay_dir)
        for i, (ov, (png_path, y_off)) in enumerate(zip(overlays, png_specs)):
            inputs += ["-i", str(png_path)]
            png_idx = input_idx
            input_idx += 1
            this_label = f"[vt{i}]"
            t_start = ov.start_s
            t_end = ov.start_s + ov.duration_s
            filter_parts.append(
                f"{prev_label}[{png_idx}:v]overlay=x=0:y={y_off}:"
                f"enable='between(t,{t_start:.3f},{t_end:.3f})'{this_label}"
            )
            prev_label = this_label
            text_overlay_count += 1

    # Brand logo overlay (corner)
    if brand is not None and brand.logo_path is not None and Path(brand.logo_path).exists():
        inputs += ["-i", str(brand.logo_path)]
        logo_idx = input_idx
        input_idx += 1
        logo_w = int(TARGET_W_OUT * (brand.logo_size_pct / 100))
        margin = 40
        pos_map = {
            "top_left": (margin, margin),
            "top_right": (TARGET_W_OUT - logo_w - margin, margin),
            "bottom_left": (margin, TARGET_H_OUT - logo_w - margin),
            "bottom_right": (TARGET_W_OUT - logo_w - margin, TARGET_H_OUT - logo_w - margin),
        }
        x, y = pos_map.get(brand.logo_position, pos_map["top_right"])
        # Pre-scale + alpha
        filter_parts.append(
            f"[{logo_idx}:v]scale={logo_w}:-1,format=rgba,"
            f"colorchannelmixer=aa={brand.logo_opacity:.3f}[logo]"
        )
        filter_parts.append(f"{prev_label}[logo]overlay=x={x}:y={y}[vout]")
        prev_label = "[vout]"
    elif text_overlay_count > 0:
        # Final text overlay should output [vout]
        # Rename last vtN to vout via an extra null pass
        filter_parts.append(f"{prev_label}null[vout]")
        prev_label = "[vout]"

    if spine_audio is not None:
        src = prev_label if (text_overlay_count or (brand and brand.logo_path)) else "[0:v]"
        fade_st = max(0.0, total_duration_s - 0.6)
        filter_parts.append(f"{src}fade=t=out:st={fade_st:.2f}:d=0.6[vfade]")
        prev_label = "[vfade]"

    video_map = prev_label if (spine_audio is not None or text_overlay_count or (brand and brand.logo_path)) else "0:v"

    filter_complex = ";".join(filter_parts)

    args = [
        "ffmpeg", "-y",
        *inputs,
        "-filter_complex", filter_complex,
        "-map", video_map,
        "-map", "[aout]",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k", "-ar", "44100", "-ac", "2",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        "-shortest",
        str(out_path),
    ]
    _run(args)


def _extract_spine_audio(edl: EDL, work_dir: Path) -> Path | None:
    if edl.audio_spine is None:
        return None
    sp = edl.audio_spine
    # The spine window may be longer than the video timeline; -shortest will cut
    # audio at the video end, so the fade must sit at the TIMELINE end or it is
    # never heard.
    dur = min(sp.out_s - sp.in_s, edl.total_duration_s)
    out = work_dir / "spine.wav"
    fade_start = max(0.0, dur - 1.4)
    _run([
        "ffmpeg", "-y", "-ss", f"{sp.in_s}", "-i", str(sp.asset_path),
        "-t", f"{dur}", "-vn", "-ac", "2", "-ar", "44100",
        "-af", f"afade=t=out:st={fade_start:.2f}:d=1.4",
        str(out),
    ])
    return out


def assemble(
    edl: EDL,
    assets: list[MediaAsset],
    out_path: Path,
    work_dir: Path | None = None,
    voiceover_path: Path | None = None,
    brand: BrandKit | None = None,
    watermark_text: str | None = None,
    normalize_audio: bool = True,
) -> Path:
    """Render the final reel."""
    work_dir = work_dir or Path(tempfile.mkdtemp(prefix="roughcut_"))
    work_dir.mkdir(parents=True, exist_ok=True)

    assets_by_path = {a.path: a for a in assets}
    crop_plans = plan_crops(edl.arc, assets_by_path)

    # Render each clip
    clip_paths: list[Path] = []
    for i, (clip, plan) in enumerate(zip(edl.arc, crop_plans)):
        asset = assets_by_path[clip.asset_path]
        clip_out = work_dir / f"clip_{i:03d}.mp4"
        _render_clip(clip, asset, plan, clip_out)
        clip_paths.append(clip_out)

    concat_out = work_dir / "concat.mp4"
    transitions = [c.transition_in for c in edl.arc]
    if any(t == "fade" for t in transitions):
        durations = [c.duration_s for c in edl.arc]
        _concat_clips_xfade(clip_paths, durations, transitions, concat_out)
    else:
        _concat_clips(clip_paths, concat_out)

    spine_audio = _extract_spine_audio(edl, work_dir)

    # Clamp overlays into the actual timeline — the story model sometimes
    # schedules a closer past the final frame, where it would never display.
    total = edl.total_duration_s
    overlays = []
    for ov in edl.overlays:
        if ov.start_s >= total - 0.3:
            ov = ov.model_copy(update={"start_s": max(0.0, total - ov.duration_s - 0.3)})
        overlays.append(ov)
    # Inject watermark overlay if requested (covers the full timeline)
    if watermark_text:
        overlays.append(
            TextOverlay(
                type="lower_third",
                text=watermark_text,
                start_s=0.0,
                duration_s=max(edl.total_duration_s, 1.0),
                style="watermark",
            )
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    _mix_audio_and_overlays(
        video_in=concat_out,
        music_path=edl.music_path,
        voiceover_path=voiceover_path,
        overlays=overlays,
        sfx_cues=edl.sfx,
        music_gain_db=edl.music_gain_db,
        out_path=out_path,
        brand=brand,
        normalize_audio=normalize_audio,
        spine_audio=spine_audio,
        total_duration_s=edl.total_duration_s,
    )
    return out_path
