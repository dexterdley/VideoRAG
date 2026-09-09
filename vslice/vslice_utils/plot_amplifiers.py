"""
plot_amplifiers.py
──────────────────
Visualises all proposed diff_attn_boost variants from the op-amp analogy.

For a dummy preference pair the base signal is:
    diff = logits_yes - logits_no   (swept from -6 to +6)

Two subplots per variant:
  Left  – Boosted output vs. diff  (compare curve shapes)
  Right – Effective gain = output / diff  (shows where amplification happens)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

matplotlib.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ── Input sweep ──────────────────────────────────────────────────────────────
diff = np.linspace(-6, 6, 1000)          # diff = logits_yes - logits_no
eps  = 1e-8                              # avoid /0 in gain computation

# ── Helper: sigmoid / tanh in numpy ─────────────────────────────────────────
def sigmoid(x):   return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))
def tanh(x):      return np.tanh(x)

# ── Amplifier definitions ─────────────────────────────────────────────────────

def identity_passthrough(d):
    """Baseline — No boost; identity pass-through."""
    return d

def try_amplifier(d):
    """Voltage Follower / IA v0 — diff * (1 + |tanh(diff/2)|)."""
    gate = np.tanh(d / 2.0)
    return d + 2.0 * gate

def tanh_amplifier(d):
    """Voltage Follower / IA v0 — diff * (1 + |tanh(diff/2)|)."""
    gate = np.tanh(d / 2.0)
    return d * (1.0 + np.abs(gate))

def quadratic_gate_amplifier(d):
    """Quadratic Gate Amplifier — diff * (1 + gate²) where gate = tanh(diff/2)."""
    gate = np.tanh(d / 2.0)
    return d * (1.0 + gate**2)

def instrumentation_amplifier(d):
    """Instrumentation Amplifier — diff * (1 + d²/(1+d²))."""
    gain = d**2 / (1.0 + d**2)
    return d * (1.0 + gain)

def butterworth_flat_amplifier(d):
    """
    Butterworth Expander — diff * (1 + d^4 / (1 + d^4)).
    Behavior: Gain is exactly 1.0x near zero with a remarkably flat slope.
    Once |d| crosses 1.0, the gain rapidly but smoothly transitions to 2.0x.
    No absolute values needed because d^4 is inherently symmetric.
    """
    d2 = d**2
    d4 = d2**2
    gain = d4 / (1.0 + d4)
    return d * (1.0 + gain)

# ── Collection ────────────────────────────────────────────────────────────────
variants = [
    ("Baseline (no boost)",                   identity_passthrough,     "gray",   "--"),
    ("Quadratic Gate Amplifier",              quadratic_gate_amplifier, "cyan",   "-"),
    ("Instrumentation Amplifier",             instrumentation_amplifier,"orange", "-"),
    ("Tanh Amplifier",                        tanh_amplifier,           "blue",   "-"),
    ("Try Amplifier",                         try_amplifier,           "brown",   "-"),
    ("Butterworth Flat Amplifier",            butterworth_flat_amplifier,"purple","-"),
]

outputs = [(name, fn(diff), color, ls) for name, fn, color, ls in variants]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(18, 11))

outer = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35,
                          left=0.06, right=0.97, top=0.90, bottom=0.08)

# ─── LEFT: Output vs diff ─────────────────────────────────────────────────────q
ax_out = fig.add_subplot(outer[0])
ax_out.axhline(0, color="black", lw=0.8)
ax_out.axvline(0, color="black", lw=0.8)

# shade borderline zone
ax_out.axvspan(-1.5, 1.5, alpha=0.10, color="magenta", label="_borderline zone")
ax_out.text(0, 11.5, "borderline zone", ha="center", color="magenta",
            fontsize=8, alpha=0.9)

for name, out, color, ls in outputs:
    ax_out.plot(diff, out, color=color, ls=ls, lw=2.0, label=name)

ax_out.set_xlim(-6, 6)
ax_out.set_ylim(-13, 13)
ax_out.set_xlabel("diff  =  logits_yes − logits_no", fontsize=11)
ax_out.set_ylabel("Boosted output", fontsize=11)
ax_out.set_title("Output shape of each amplifier", fontsize=13, pad=10)
leg = ax_out.legend(loc="upper left", fontsize=8.5, framealpha=0.85)

# ─── RIGHT: Effective Gain vs diff ───────────────────────────────────────────
ax_gain = fig.add_subplot(outer[1])
ax_gain.axhline(1, color="black", lw=0.8, ls="--")   # gain = 1 reference
ax_gain.axvline(0, color="black", lw=0.8)

ax_gain.axvspan(-1.5, 1.5, alpha=0.10, color="magenta")
ax_gain.text(0, 3.12, "borderline zone", ha="center", color="magenta",
             fontsize=8, alpha=0.9)

for name, out, color, ls in outputs:
    gain = out / (diff + eps)
    # smooth singularity artefacts near diff=0
    gain = np.where(np.abs(diff) < 0.05, np.nan, gain)
    lbl = name.split("\n")[0]      # single-line label for gain plot
    ax_gain.plot(diff, gain, color=color, ls=ls, lw=2.0, label=lbl)

ax_gain.set_xlim(-6, 6)
ax_gain.set_ylim(0, 3.2)
ax_gain.set_xlabel("diff  =  logits_yes − logits_no", fontsize=11)
ax_gain.set_ylabel("Effective gain  =  output / diff", fontsize=11)
ax_gain.set_title("Effective gain — where each amp amplifies", fontsize=13, pad=10)
ax_gain.legend(loc="upper right", fontsize=8.5, framealpha=0.85)

# ─── Annotations ─────────────────────────────────────────────────────────────
ann_kw = dict(fontsize=8.5, color="black", ha="center")
for ax in (ax_out, ax_gain):
    ax.text(-4.5, ax.get_ylim()[0] + 0.5, "← clear rejected", **ann_kw)
    ax.text( 4.5, ax.get_ylim()[0] + 0.5, "clear chosen →", **ann_kw)

fig.suptitle(
    "diff_attn_boost variants  —  op-amp analogy for DPO preference logits\n"
    "TIA, Log Amp, PLL expand borderline pairs; IA, Schmitt & Voltage Follower expand clear pairs",
    fontsize=13, y=0.96
)

plt.savefig("amplifier_comparison.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
print("Saved -> amplifier_comparison.png")
plt.show()
