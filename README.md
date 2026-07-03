<p align="center"><img src="assets/logo_black_1024.png#gh-light-mode-only" width="96"/><img src="assets/logo_white_1024.png#gh-dark-mode-only" width="96"/></p>

# roughcut

*Every camera roll has a diamond in it.*

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

Apple Memories is slideshow-grade. Opus Clip splits one long video into many shorts (the opposite problem). CapCut templates look auto-generated. Nothing turns a folder of raw event footage into something you'd actually post - so people either spend hours in an editor or post nothing.

## How it works

```
INGEST       ffprobe + EXIF: metadata, rotation-aware dimensions
SPEECH       Whisper (mlx, local): ground-truth transcripts per clip
UNDERSTAND   Gemini: per-clip scores, best moments, shot type (A-roll/B-roll), audio analysis
STORY        Claude: Edit Decision List - clip order, per-clip audio decision,
             overlays, SFX cues, music mood. best_moments are binding.
RENDER       ffmpeg: crop/punch-in/Ken Burns, xfade, music ducked under speech,
             synthesized SFX, PIL text overlays, -14 LUFS loudnorm
```

The EDL (edit decision list) is a JSON file you can inspect, hand-edit, or mutate conversationally:

```bash
roughcut edit "trim the close by a second and add a boom at 0:05"
```

## Two modes

**Casual** - zero config. Auto-detects the event (concert / trip / party / wedding), invents the narrative, leans on real audio from the scene.

```bash
roughcut casual ~/Photos/tokyo-trip --music track.mp3
```

**Pro** - for creators. Brief-driven, with reference-reel style transfer, brand kits, script alignment (your talking-head A-roll is matched to your script via Whisper, B-roll cut in between), and multi-variant generation.

```bash
roughcut pro ~/footage/ep12 \
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
roughcut init-sfx        # generate the synthesized SFX library
```

Cost: roughly $0.50–1.00 in API calls per reel (Gemini video analysis dominates). Whisper and rendering run locally.

## Commands

| Command | What |
|---|---|
| `roughcut casual DIR` | folder → reel, zero config |
| `roughcut pro DIR --brief "..."` | creator mode: script, refs, brand, variants |
| `roughcut edit "..."` | natural-language edits to the last cut |
| `roughcut ingest / speech / understand / story / render` | run stages individually (cached JSON between stages) |
| `roughcut brand-init` | scaffold a brand kit |
| `roughcut init-sfx` | regenerate SFX wavs |

## Testing

```bash
python -m pytest tests/                 # 30 unit tests, no API needed
python scripts/make_test_fixtures.py    # synthetic event folder (TTS speech,
                                        # rotated clip, silent clip, photos)
python scripts/make_test_track.py       # 120 BPM test track
roughcut pro /tmp/roughcut_fixtures --brief "test" \
  --script /tmp/roughcut_fixtures/script.txt --music /tmp/test_track.wav
```

## Roadmap

See [ROADMAP.md](ROADMAP.md). Highlights: incremental re-render cache for the edit loop, auto music, event templates, iOS Shortcuts, Mac app, fully-local model path (Qwen-VL + MLX).
