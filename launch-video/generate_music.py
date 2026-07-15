from __future__ import annotations

import math
import wave
from pathlib import Path

import numpy as np

SAMPLE_RATE = 48_000
DURATION = 72.0
BPM = 124.0
BEAT = 60.0 / BPM
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "output" / "music-original.wav"
RNG = np.random.default_rng(258)


def hz(midi: float) -> float:
    return 440.0 * 2.0 ** ((midi - 69.0) / 12.0)


def add_tone(track: np.ndarray, start: float, duration: float, freq: float, amp: float,
             pan: float = 0.0, kind: str = "sine", release: float = 0.25) -> None:
    i0 = max(0, int(start * SAMPLE_RATE))
    n = min(int(duration * SAMPLE_RATE), len(track) - i0)
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    phase = 2 * np.pi * freq * t
    if kind == "saw":
        sig = 2.0 * ((freq * t) % 1.0) - 1.0
        sig = 0.72 * sig + 0.28 * np.sin(phase)
    elif kind == "square":
        sig = np.tanh(2.2 * np.sin(phase))
    elif kind == "pluck":
        sig = np.sin(phase) + 0.28 * np.sin(phase * 2.01) + 0.12 * np.sin(phase * 3.0)
    else:
        sig = np.sin(phase)
    attack = max(1, int(min(0.018, duration / 4) * SAMPLE_RATE))
    rel = max(1, int(min(release, duration / 2) * SAMPLE_RATE))
    env = np.ones(n, dtype=np.float32)
    env[:attack] = np.linspace(0, 1, attack, dtype=np.float32)
    env[-rel:] *= np.linspace(1, 0, rel, dtype=np.float32)
    if kind == "pluck":
        env *= np.exp(-3.8 * t / max(duration, 0.01))
    left = math.sqrt((1.0 - pan) * 0.5)
    right = math.sqrt((1.0 + pan) * 0.5)
    track[i0:i0+n, 0] += sig * env * amp * left
    track[i0:i0+n, 1] += sig * env * amp * right


def add_kick(track: np.ndarray, start: float, amp: float = 0.58) -> None:
    n = int(0.42 * SAMPLE_RATE)
    i0 = int(start * SAMPLE_RATE)
    n = min(n, len(track) - i0)
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    phase = 2 * np.pi * (44 * t + 82 * 0.045 * (1 - np.exp(-t / 0.045)))
    sig = np.sin(phase) * np.exp(-t * 12.5)
    sig += 0.18 * RNG.normal(0, 1, n) * np.exp(-t * 45)
    track[i0:i0+n] += (sig * amp)[:, None]


def add_snare(track: np.ndarray, start: float, amp: float = 0.18) -> None:
    n = int(0.28 * SAMPLE_RATE)
    i0 = int(start * SAMPLE_RATE)
    n = min(n, len(track) - i0)
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = RNG.normal(0, 1, n).astype(np.float32)
    tone = np.sin(2 * np.pi * 185 * t)
    sig = (0.78 * noise + 0.22 * tone) * np.exp(-t * 18)
    track[i0:i0+n, 0] += sig * amp * 0.92
    track[i0:i0+n, 1] += sig * amp * 1.08


def add_hat(track: np.ndarray, start: float, amp: float = 0.055, pan: float = 0.0) -> None:
    n = int(0.075 * SAMPLE_RATE)
    i0 = int(start * SAMPLE_RATE)
    n = min(n, len(track) - i0)
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = RNG.normal(0, 1, n).astype(np.float32)
    # Difference filtering keeps the hat bright without another DSP dependency.
    noise[1:] = noise[1:] - noise[:-1]
    sig = noise * np.exp(-t * 55)
    left = math.sqrt((1.0 - pan) * 0.5)
    right = math.sqrt((1.0 + pan) * 0.5)
    track[i0:i0+n, 0] += sig * amp * left
    track[i0:i0+n, 1] += sig * amp * right


def add_impact(track: np.ndarray, start: float, amp: float = 0.35) -> None:
    n = int(1.2 * SAMPLE_RATE)
    i0 = int(start * SAMPLE_RATE)
    n = min(n, len(track) - i0)
    if n <= 0:
        return
    t = np.arange(n, dtype=np.float32) / SAMPLE_RATE
    noise = RNG.normal(0, 1, n).astype(np.float32)
    sig = (0.65 * np.sin(2 * np.pi * 47 * t) + 0.2 * noise) * np.exp(-t * 4.4)
    track[i0:i0+n] += (sig * amp)[:, None]


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    track = np.zeros((int(DURATION * SAMPLE_RATE), 2), dtype=np.float32)
    chords = [
        (50, 53, 57, 60),  # Dm7
        (46, 50, 53, 57),  # Bbmaj7
        (41, 45, 48, 53),  # F
        (48, 52, 55, 60),  # C
    ]

    # Low pulse arrives after the opening hook and builds in three acts.
    total_beats = int(DURATION / BEAT)
    for beat_idx in range(total_beats):
        t0 = beat_idx * BEAT
        chord = chords[(beat_idx // 8) % len(chords)]
        energy = 0.0 if t0 < 4 else (0.65 if t0 < 20 else 1.0)
        if t0 > 60:
            energy *= max(0.25, 1.0 - (t0 - 60) / 15)
        if energy <= 0:
            continue

        if beat_idx % 4 in (0, 2):
            add_kick(track, t0, 0.38 * energy)
        if beat_idx % 4 in (1, 3) and t0 > 12:
            add_snare(track, t0, 0.12 * energy)
        if t0 > 7:
            add_hat(track, t0, 0.030 * energy, -0.25 if beat_idx % 2 else 0.25)
            add_hat(track, t0 + BEAT / 2, 0.024 * energy, 0.30 if beat_idx % 2 else -0.30)

        root = chord[0] - 12
        if beat_idx % 2 == 0:
            add_tone(track, t0, BEAT * 1.7, hz(root), 0.12 * energy, kind="saw", release=0.12)

        # Lilac glass arpeggio: restrained, legible under narration.
        note = chord[beat_idx % len(chord)] + 12
        pan = [-0.55, 0.45, -0.18, 0.62][beat_idx % 4]
        add_tone(track, t0 + BEAT * 0.08, BEAT * 0.82, hz(note), 0.050 * energy,
                 pan=pan, kind="pluck", release=0.20)

        if beat_idx % 8 == 0 and t0 > 18:
            for n_idx, midi in enumerate(chord):
                add_tone(track, t0, BEAT * 7.4, hz(midi + 12), 0.020 * energy,
                         pan=-0.55 + n_idx * 0.36, kind="sine", release=0.9)

    for cue in (5.8, 14.0, 23.0, 32.0, 43.0, 51.0, 60.0, 67.0):
        add_impact(track, cue, 0.24 if cue < 60 else 0.18)

    # Simple stereo delay/reverb, kept short so the edit remains punchy.
    for delay_s, gain, swap in ((0.17, 0.16, True), (0.31, 0.10, False), (0.47, 0.055, True)):
        d = int(delay_s * SAMPLE_RATE)
        source = track[:-d, ::-1] if swap else track[:-d]
        track[d:] += source * gain

    # Gentle master saturation and headroom for voiceover.
    track = np.tanh(track * 1.15)
    peak = float(np.max(np.abs(track))) or 1.0
    track *= 0.72 / peak
    pcm = np.int16(np.clip(track, -1, 1) * 32767)
    with wave.open(str(OUT), "wb") as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())
    print(OUT)


if __name__ == "__main__":
    main()
