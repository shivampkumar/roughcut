<p align="center"><img src="assets/logo_black_1024.png#gh-light-mode-only" width="96"/><img src="assets/logo_white_1024.png#gh-dark-mode-only" width="96"/></p>

# roughcut

*Every camera roll has a diamond in it.*

Point it at a folder of raw clips and photos. It watches every clip, finds the moments, keeps the audio that matters, cuts on the beat, and hands you a finished vertical reel.

<table>
<tr>
<td width="38%">
<img src="assets/demo.gif" width="100%"/>
<sub>Input: a stranger's real camera roll from a fireworks festival in Nagano (5 handheld clips + 6 photos, <a href="examples/fireworks/ATTRIBUTION.md">CC BY-SA, Wikimedia Commons</a>). Output: this 20-second reel, zero human editing. <a href="examples/fireworks/demo_reel.mp4">Full quality with audio</a>.</sub>
</td>
<td>

```jsonc
// what the AI actually produces: an EDIT DECISION
// LIST, not a black-box video.
// full file: examples/fireworks/edl.json
{
  "event_type": "casual",
  "arc": [
    { "asset_path": "sanada_fireworks_1.mp4",
      "in_s": 4.0, "out_s": 7.0,
      "role": "hook", "audio_decision": "keep_loud" },
    { "asset_path": "sanada_fireworks_photo_5.jpg",
      "role": "energy", "punch_in": true },
    { "asset_path": "sanada_fireworks_3.mp4",
      "in_s": 25.0, "out_s": 28.5,
      "role": "close", "transition_in": "fade" }
  ],
  "overlays": [
    { "type": "hook",
      "text": "october nights in nagano",
      "start_s": 0.3 }
  ]
}
```

</td>
</tr>
</table>

## The design bet: an inspectable edit, not a generated video

Most AI video tools give you an opaque render. roughcut's story stage outputs an **edit decision list**, the same artifact a human editor produces. That means you can:

- **read it**: see exactly why each cut exists, which 3 seconds of each clip were chosen, and what happens to each clip's audio
- **change it by talking**: `roughcut edit "trim the close by a second and add a boom at 0:05"` parses to structured edit ops and re-renders
- **change it by hand**: it's JSON; edit and re-render without re-paying for analysis

The other opinionated part is **audio**. Every clip gets a decision: meaningful speech stays loud with music ducked under it, crowd noise stays natural, wind gets muted, barely-audible speech becomes subtitles. Reels fail on audio more than video, and most tools just pave everything with music.

## Try it with zero API keys

The repo ships the example's cached analysis + EDL. You only need the source footage:

```bash
uv venv .venv && source .venv/bin/activate && uv pip install -e .
bash scripts/get_demo_footage.sh     # ~130MB, Wikimedia Commons
roughcut render --edl examples/fireworks/edl.json \
  --assets examples/fireworks/assets.json --out reel.mp4
```

That runs the real render stage (crop, punch-ins, fades, SFX, loudnorm) on your machine. No accounts, no keys.

## Run it on your own footage

Needs `GEMINI_API_KEY` + `ANTHROPIC_API_KEY` in `.env` (roughly $0.50-1.00 of API per reel; Whisper and rendering are local). macOS Apple Silicon.

```bash
cp .env.example .env   # add keys
roughcut casual ~/Photos/that-one-night --out reel.mp4
```

Creator mode adds a brief, script alignment, reference-style transfer, brand kits, and variants:

```bash
roughcut pro ~/footage/ep12 \
  --brief "cold open on the reveal, hyped but clean" \
  --script script.txt --ref refs/ep10.mp4 --brand brand.json --variants 3
```

With `--script`, your talking-head A-roll is matched to your script line-by-line via Whisper, and B-roll is cut between beats.

## How it works

```
INGEST       ffprobe + EXIF: metadata, rotation-aware dimensions
SPEECH       Whisper (mlx, local): ground-truth transcripts
UNDERSTAND   Gemini: per-clip scores, best moments, shot type, audio analysis
STORY        Claude: the EDL. best_moments are binding; the model may only
             cut inside windows the analysis pass judged worth using
RENDER       ffmpeg: smart 9:16 crop (face-tracked), punch-ins, Ken Burns,
             xfade, music ducked under speech, synthesized SFX, PIL text
             overlays, -14 LUFS loudnorm
```

Every stage caches JSON, so you can re-run any stage alone (`roughcut story`, `roughcut render`) and iterate on the cut without re-paying for analysis.

## Testing without footage

```bash
python -m pytest tests/                  # 31 unit tests, no API
python scripts/make_test_fixtures.py     # synthetic event: TTS speech clip,
                                         # rotated clip, silent clip, photos
```

## Roadmap

See [ROADMAP.md](ROADMAP.md). Next up: incremental re-render cache for the edit loop, auto music, event templates, and a fully local model path.

## Demo footage credit

Fireworks footage and photos by [KENPEI](https://commons.wikimedia.org/wiki/User:KENPEI), Wikimedia Commons, CC BY-SA 4.0. The example reel is a derivative work under the same license. Details in [ATTRIBUTION.md](examples/fireworks/ATTRIBUTION.md).
