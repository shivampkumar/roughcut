"""Synthesized SFX library.

Generates 8 standard SFX wavs deterministically from numpy. Zero licensing,
ships with the package. Re-run `montage init-sfx` to regenerate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

SR = 44100  # 44.1 kHz
ASSETS_DIR = Path(__file__).parent / "assets" / "sfx"


def _envelope(n: int, attack: float = 0.005, release: float = 0.08) -> np.ndarray:
    """ADSR-ish envelope. Attack/release in seconds."""
    a = int(attack * SR)
    r = int(release * SR)
    s = max(0, n - a - r)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.ones(s),
        np.linspace(1, 0, r),
    ])
    if len(env) < n:
        env = np.pad(env, (0, n - len(env)))
    return env[:n]


def _normalize(x: np.ndarray, target_db: float = -3.0) -> np.ndarray:
    peak = np.max(np.abs(x))
    if peak < 1e-9:
        return x
    target = 10 ** (target_db / 20)
    return x * (target / peak)


# ---------- generators ----------

def whoosh(dur: float = 0.45) -> np.ndarray:
    """Filtered noise sweep, low → high → low. Transition SFX."""
    n = int(dur * SR)
    noise = np.random.uniform(-1, 1, n)
    # Cheap "filter sweep": modulate amplitude with a sine to fake spectral motion,
    # plus an exponential pitch-style frequency emphasis via simple IIR.
    t = np.linspace(0, dur, n)
    # Two-stage low-pass with time-varying alpha (small = more LP)
    alpha = 0.05 + 0.40 * np.exp(-((t - dur / 2) ** 2) / (2 * (dur / 4) ** 2))
    y = np.empty(n)
    last = 0.0
    for i in range(n):
        last = (1 - alpha[i]) * last + alpha[i] * noise[i]
        y[i] = last
    env = _envelope(n, attack=0.03, release=0.20)
    return _normalize(y * env)


def impact(dur: float = 0.18) -> np.ndarray:
    """Kick-drum-style hit. Beat punctuation."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n)
    # Pitched body: sine sweeping from 110 Hz to 45 Hz
    freq = 110 * np.exp(-t * 18) + 45
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase)
    # Click transient
    click = np.exp(-t * 200) * np.random.uniform(-1, 1, n) * 0.3
    env = np.exp(-t * 10)
    return _normalize(body * env + click)


def pop(dur: float = 0.08) -> np.ndarray:
    """Short bright burst. Text-appear punctuation."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n)
    freq = 1200 * np.exp(-t * 30) + 400
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) + 0.4 * np.sin(2 * phase)
    env = np.exp(-t * 35)
    return _normalize(body * env)


def shimmer(dur: float = 0.6) -> np.ndarray:
    """High frequency sparkle. Photo zoom-in / magic feel."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n)
    # Sum of high-frequency sines with random phases
    rng = np.random.default_rng(seed=7)
    freqs = rng.uniform(2400, 7000, 12)
    phases = rng.uniform(0, 2 * np.pi, 12)
    y = np.zeros(n)
    for f, p in zip(freqs, phases):
        y += np.sin(2 * np.pi * f * t + p) * rng.uniform(0.3, 1.0)
    # Bell envelope
    env = np.sin(np.pi * t / dur) ** 1.5
    return _normalize(y * env / len(freqs))


def click(dur: float = 0.03) -> np.ndarray:
    """Sharp transient. Text snap."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n)
    body = np.sin(2 * np.pi * 3000 * t)
    env = np.exp(-t * 200)
    return _normalize(body * env)


def boom(dur: float = 0.7) -> np.ndarray:
    """Heavy low end. Climax emphasis."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n)
    freq = 80 * np.exp(-t * 4) + 35
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) + 0.4 * np.sin(2 * phase)
    rumble = 0.15 * np.random.uniform(-1, 1, n)
    env = np.exp(-t * 3)
    return _normalize((body + rumble) * env, target_db=-2.0)


def swell(dur: float = 1.5) -> np.ndarray:
    """Rising build before a drop. Pre-climax."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n)
    noise = np.random.uniform(-1, 1, n)
    # Increasing low-pass cutoff: high → wider band → "opening up"
    alpha = 0.02 + 0.50 * (t / dur)
    y = np.empty(n)
    last = 0.0
    for i in range(n):
        last = (1 - alpha[i]) * last + alpha[i] * noise[i]
        y[i] = last
    env = (t / dur) ** 1.8
    return _normalize(y * env)


def riser(dur: float = 1.2) -> np.ndarray:
    """Pitched tonal riser. Tension build."""
    n = int(dur * SR)
    t = np.linspace(0, dur, n)
    freq = 200 * np.exp(t * 2.5)  # exponential sweep
    phase = 2 * np.pi * np.cumsum(freq) / SR
    body = np.sin(phase) + 0.3 * np.sin(2 * phase)
    env = (t / dur) ** 1.5
    return _normalize(body * env)


_GENERATORS = {
    "whoosh": whoosh,
    "impact": impact,
    "pop": pop,
    "shimmer": shimmer,
    "click": click,
    "boom": boom,
    "swell": swell,
    "riser": riser,
}


def generate_all(out_dir: Path = ASSETS_DIR) -> list[Path]:
    """Generate every SFX wav. Returns paths written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, fn in _GENERATORS.items():
        audio = fn()
        # Stereo by duplicating
        stereo = np.stack([audio, audio], axis=1)
        path = out_dir / f"{name}.wav"
        sf.write(str(path), stereo, SR, subtype="PCM_16")
        paths.append(path)
    return paths


if __name__ == "__main__":
    paths = generate_all()
    for p in paths:
        print(p)
