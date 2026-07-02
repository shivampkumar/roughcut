# montage

Camera roll → finished Reel. Drop a folder of raw clips and photos, get back one edited short with music, captions, and cuts that land on the beat.

```
input/                                output/reel.mp4
  IMG_0001.jpg                          1080x1920, 30fps
  IMG_0002.jpg          ──────▶         beat-snapped cuts
  VID_001.mp4          one command      smart 9:16 crop
  VID_002.mp4                           speech kept loud, noise muted
  ...                                   text overlays, SFX, loudnorm
```

## Why

Apple Memories is slideshow-grade. Opus Clip splits one long video into many shorts (the opposite problem). CapCut templates look auto-generated. Nothing turns a folder of raw event footage into something you'd actually post — so people either spend hours in an editor or post nothing.

## How it works

```
INGEST       ffprobe + EXIF: metadata, rotation-aware dimensions
SPEECH       Whisper (mlx, local): ground-truth transcripts per clip
UNDERSTAND   Gemini: per-clip scores, best moments, shot type (A-roll/B-roll), audio analysis
STORY        Claude: Edit Decision List — clip order, per-clip audio decision,
             overlays, SFX cues, music mood. best_moments are binding.
RENDER       ffmpeg: crop/punch-in/Ken Burns, xfade, music ducked under speech,
             synthesized SFX, PIL text overlays, -14 LUFS loudnorm
```

The EDL (edit decision list) is a JSON file you can inspect, hand-edit, or mutate conversationally:

```bash
montage edit "trim the close by a second and add a boom at 0:05"
```

## Two modes

**Casual** — zero config. Auto-detects the event (concert / trip / party / wedding), invents the narrative, leans on real audio from the scene.

```bash
montage casual ~/Photos/tokyo-trip --music track.mp3
```

**Pro** — for creators. Brief-driven, with reference-reel style transfer, brand kits, script alignment (your talking-head A-roll is matched to your script via Whisper, B-roll cut in between), and multi-variant generation.

```bash
montage pro ~/footage/ep12 \
  --brief "cold open on the reveal, hyped but clean" \
  --script script.txt \
  --ref refs/ep10.mp4 \
  --brand brand.json \
  --variants 3
```

## Setup

Requires: macOS (Apple Silicon), Python ≥ 3.11, ffmpeg (`brew install ffmpeg`).

```bash
uv venv .venv && source .venv/bin/activate
uv pip install -e .
cp .env.example .env    # add GEMINI_API_KEY + ANTHROPIC_API_KEY
montage init-sfx        # generate the synthesized SFX library
```

Cost: roughly $0.50–1.00 in API calls per reel (Gemini video analysis dominates). Whisper and rendering run locally.

## Commands

| Command | What |
|---|---|
| `montage casual DIR` | folder → reel, zero config |
| `montage pro DIR --brief "..."` | creator mode: script, refs, brand, variants |
| `montage edit "..."` | natural-language edits to the last cut |
| `montage ingest / speech / understand / story / render` | run stages individually (cached JSON between stages) |
| `montage brand-init` | scaffold a brand kit |
| `montage init-sfx` | regenerate SFX wavs |

## Testing

```bash
python -m pytest tests/                 # 30 unit tests, no API needed
python scripts/make_test_fixtures.py    # synthetic event folder (TTS speech,
                                        # rotated clip, silent clip, photos)
python scripts/make_test_track.py       # 120 BPM test track
montage pro /tmp/montage_fixtures --brief "test" \
  --script /tmp/montage_fixtures/script.txt --music /tmp/test_track.wav
```

## Roadmap

See [plan.md](plan.md). Highlights: incremental re-render cache for the edit loop, auto music, event templates, iOS Shortcuts, Mac app, fully-local model path (Qwen-VL + MLX).
