"""Synthesize a 32s, 120 BPM test track. Exercises beat detection, snapping,
ducking, and loudnorm without licensing anything."""

import numpy as np
import soundfile as sf

SR = 44100
BPM = 120
BEAT = 60 / BPM
DUR = 32.0

n_total = int(DUR * SR)
audio = np.zeros(n_total)


def _add(at_s: float, sig: np.ndarray) -> None:
    n0 = int(at_s * SR)
    n1 = min(n_total, n0 + len(sig))
    if n0 < n_total:
        audio[n0:n1] += sig[: n1 - n0]


# Kick on every beat
kick_t = np.arange(int(0.15 * SR)) / SR
kick_freq = 100 * np.exp(-kick_t * 25) + 40
kick = np.sin(2 * np.pi * np.cumsum(kick_freq) / SR) * np.exp(-kick_t * 12) * 0.9
for b in np.arange(0, DUR, BEAT):
    _add(b, kick)

# Hat on offbeats
rng = np.random.default_rng(0)
hat_len = int(0.03 * SR)
for b in np.arange(BEAT / 2, DUR, BEAT):
    hat = rng.uniform(-1, 1, hat_len) * np.exp(-np.arange(hat_len) / SR * 200) * 0.22
    _add(b, hat)

# Bass progression, two beats per note
prog = [55.0, 55.0, 65.4, 49.0]  # A A C G
note_len = int(BEAT * 2 * SR * 0.9)
note_t = np.arange(note_len) / SR
for i, b in enumerate(np.arange(0, DUR, BEAT * 2)):
    f = prog[i % len(prog)]
    env = np.minimum(1, note_t * 10) * np.exp(-note_t * 1.5)
    _add(b, 0.3 * np.sin(2 * np.pi * f * note_t) * env)

# Simple energy lift halfway (volume ramp = fake "drop")
ramp = np.ones(n_total)
half = n_total // 2
ramp[half - SR : half] = np.linspace(1.0, 0.3, SR)   # duck before drop
ramp[half : half + SR] = np.linspace(1.3, 1.0, SR)   # hit after
audio *= ramp

audio /= np.max(np.abs(audio))
stereo = np.stack([audio * 0.8, audio * 0.8], axis=1)
sf.write("/tmp/test_track.wav", stereo, SR)
print("/tmp/test_track.wav written:", DUR, "s @", BPM, "BPM")
