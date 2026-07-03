# roughcut roadmap

## Shipped (v0.1)

- Five-stage pipeline: ingest (rotation-aware) -> Whisper speech -> Gemini per-clip analysis -> Claude EDL -> ffmpeg render
- Casual mode (zero config) and pro mode (brief, script alignment, reference style transfer, brand kits, multi-variant)
- Conversational editing: `roughcut edit "trim the close by a second"`
- Per-clip audio decisions (keep speech loud, duck music, mute noise)
- Beat-snapped cuts, punch-ins, Ken Burns on photos, crossfades, synthesized SFX, -14 LUFS loudnorm
- 31 unit tests + synthetic-fixture E2E harness (no footage needed to test)

## Next (v0.2)

- [ ] Incremental re-render cache: `roughcut edit` should only re-render changed clips (content-addressed per-clip cache)
- [ ] `roughcut undo` / EDL versioning
- [ ] Smart aspect: `--aspect auto` picks 16:9 when most sources are landscape
- [ ] Auto music for casual mode (mood-matched CC0 library)
- [ ] Two-pass loudnorm for tighter loudness targets on transient-heavy audio
- [ ] Stricter aesthetic scoring rubric; penalize darkness and motion blur harder

## Later

- Event templates (concert / trip / wedding / party) with tuned arcs
- Music generation integration
- Brand LUT + intro/outro stings + voice-profile voiceover in pro mode
- iOS Shortcuts entry point
- Mac app shell (drag folder -> preview -> export)
- Fully local model path (local VLM for understanding, local LLM for story, MLX) - zero cloud, zero per-reel cost

## Contributing

Issues and PRs welcome. Run the test suite (`pytest tests/`) and the fixture E2E (`scripts/make_test_fixtures.py`) before submitting.
