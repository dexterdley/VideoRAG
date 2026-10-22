"""
Tiny procedural background-music generator (numpy only, royalty free).

Each mood is a tempo + key + chord progression + which instruments play.
The track is rendered bar-by-bar to exactly the requested duration with a
fade in/out, so no external audio assets are needed.

    python music.py --mood upbeat --duration 20 --out upbeat.wav
"""

import argparse
import wave

import numpy as np

SR = 44100

# Semitone offsets from the key root for each chord (triads + octave root).
MAJOR = {"I": (0, 4, 7), "ii": (2, 5, 9), "IV": (5, 9, 12), "V": (7, 11, 14), "vi": (9, 12, 16)}
MINOR = {"i": (0, 3, 7), "III": (3, 7, 10), "iv": (5, 8, 12), "VI": (8, 12, 15), "VII": (10, 14, 17)}

MOODS = {
    #           bpm  root(midi) chords                                  drums   arp    four_on_floor
    "upbeat":    dict(bpm=118, root=60, prog=[MAJOR[c] for c in ("I", "V", "vi", "IV")], drums=True, arp=True, four=False),
    "energetic": dict(bpm=128, root=57, prog=[MINOR[c] for c in ("i", "VI", "III", "VII")], drums=True, arp=True, four=True),
    "chill":     dict(bpm=84,  root=62, prog=[MAJOR[c] for c in ("ii", "V", "I", "vi")], drums=True, arp=False, four=False),
    "epic":      dict(bpm=96,  root=50, prog=[MINOR[c] for c in ("i", "VI", "III", "VII")], drums=True, arp=False, four=False),
}
MOOD_NAMES = list(MOODS)


def midi_hz(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def envelope(n, attack, release):
    env = np.ones(n)
    a, r = min(n, int(attack * SR)), min(n, int(release * SR))
    if a:
        env[:a] = np.linspace(0, 1, a)
    if r:
        env[-r:] *= np.linspace(1, 0, r)
    return env


def tone(freq, dur, harmonics=(1.0, 0.5, 0.25), decay=None, attack=0.01, release=0.05):
    n = int(dur * SR)
    t = np.arange(n) / SR
    sig = sum(a * np.sin(2 * np.pi * freq * (k + 1) * t) for k, a in enumerate(harmonics))
    if decay:
        sig *= np.exp(-t / decay)
    return sig * envelope(n, attack, release)


def kick(dur=0.35):
    t = np.arange(int(dur * SR)) / SR
    freq = 50 + 90 * np.exp(-t * 35)
    return np.sin(2 * np.pi * np.cumsum(freq) / SR) * np.exp(-t * 9)


def noise_hit(dur, decay, highpass, rng):
    n = int(dur * SR)
    x = rng.standard_normal(n)
    if highpass:  # crude one-pole high-pass for hats
        x = np.diff(x, prepend=0.0)
    return x * np.exp(-np.arange(n) / SR / decay)


def add(buf, sig, at):
    i = int(at * SR)
    if i >= len(buf):
        return
    j = min(len(buf), i + len(sig))
    buf[i:j] += sig[: j - i]


def render(mood="upbeat", duration=20.0, seed=0):
    """Return a (n_samples, 2) float32 stereo track in [-1, 1]."""
    cfg = MOODS[mood]
    rng = np.random.default_rng(seed)
    beat = 60.0 / cfg["bpm"]
    bar = 4 * beat
    n = int(duration * SR)
    pad, bass, drums, lead = (np.zeros(n) for _ in range(4))

    n_bars = int(np.ceil(duration / bar))
    for b in range(n_bars):
        chord = cfg["prog"][b % len(cfg["prog"])]
        t0 = b * bar
        # Pad: sustained chord one octave below root, soft harmonics.
        for semi in chord:
            add(pad, tone(midi_hz(cfg["root"] - 12 + semi), bar + 0.1, (1.0, 0.3, 0.1), attack=0.15, release=0.3), t0)
        # Bass: root on every beat (8ths for energetic).
        step = beat / 2 if cfg["four"] else beat
        for k in range(int(round(bar / step))):
            add(bass, tone(midi_hz(cfg["root"] - 24 + chord[0]), step * 0.9, (1.0, 0.4), decay=0.25), t0 + k * step)
        # Drums.
        if cfg["drums"]:
            for q in range(4):
                if cfg["four"] or q in (0, 2):
                    add(drums, kick(), t0 + q * beat)
                if q in (1, 3):
                    add(drums, 0.5 * noise_hit(0.2, 0.06, False, rng), t0 + q * beat)
                for h in (0, 0.5):
                    add(drums, 0.12 * noise_hit(0.05, 0.015, True, rng), t0 + (q + h) * beat)
        # Arpeggio: chord tones on 16ths, plucky.
        if cfg["arp"]:
            notes = [chord[0], chord[1], chord[2], chord[1] + 12]
            for k in range(16):
                semi = notes[k % len(notes)]
                add(lead, tone(midi_hz(cfg["root"] + 12 + semi), beat / 4, (1.0, 0.2), decay=0.08), t0 + k * beat / 4)

    left = 0.22 * pad + 0.35 * bass + 0.45 * drums + 0.12 * lead
    right = 0.22 * pad + 0.35 * bass + 0.45 * drums + 0.18 * lead  # slight stereo spread on the arp
    mix = np.stack([left, right], axis=1)
    mix = np.tanh(1.2 * mix)  # soft clip glue
    mix *= envelope(n, 1.0, 2.0)[:, None]
    peak = np.abs(mix).max()
    if peak > 0:
        mix *= 0.89 / peak  # ~ -1 dBFS
    return mix.astype(np.float32)


def write_wav(path, audio):
    pcm = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(pcm.shape[1])
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mood", choices=MOOD_NAMES, default="upbeat")
    ap.add_argument("--duration", type=float, default=20.0)
    ap.add_argument("--out", default="bgm.wav")
    a = ap.parse_args()
    print(write_wav(a.out, render(a.mood, a.duration)))
