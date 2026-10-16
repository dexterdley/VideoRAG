# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 16:25:32 2025

@author: User
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.spatial import ConvexHull
from scipy import interpolate

root = "./downloads/"
def moving_average(data, smoothing_weight=0.99, start=0):
    """Calculates the exponentially weighted moving average."""
    smoothed = []
    last = data.iloc[0] if hasattr(data, 'iloc') else data[start]  # Initialize with the first value
    for point in data:
        smoothed_val = (1 - smoothing_weight) * point + smoothing_weight * last
        smoothed.append(smoothed_val)
        last = smoothed_val
    return np.array(smoothed)

# File paths
file_paths_kl = {
    'MiniCPM DPO': root + "run-vslice_DPO_no_margin_summe_0_20260909_163954-tag-Train_step_kl_drift.csv",
    'MiniCPM DPO + Margin': root + "run-vslice_DPO_summe_0_20260905_104913-tag-Train_step_kl_drift.csv",
    # 'MiniCPM Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_kl_drift.csv",
    'QWEN2-VL': root + "run-vslice_qwen2_vl_3b_DPO_no_margin_summe_0_20260921_181616-tag-Train_step_kl_drift.csv",
    'QWEN2-VL DPO + Margin': root + "run-vslice_qwen2_vl_3b_DPO_summe_0_20260917_053526-tag-Train_step_kl_drift.csv"
}

file_paths_loss = {
    'MiniCPM DPO': root + "run-vslice_DPO_no_margin_summe_0_20260909_163954-tag-Train_step_loss.csv",
    'MiniCPM DPO + Margin': root + "run-vslice_DPO_summe_0_20260905_104913-tag-Train_step_loss.csv",
    # 'MiniCPM Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_loss.csv",
    'QWEN2-VL': root + "run-vslice_qwen2_vl_3b_DPO_no_margin_summe_0_20260921_181616-tag-Train_step_loss.csv",
    'QWEN2-VL DPO + Margin': root + "run-vslice_qwen2_vl_3b_DPO_summe_0_20260917_053526-tag-Train_step_loss.csv",
}

file_paths_pi_ratio = {
    'MiniCPM DPO': root + "run-vslice_DPO_no_margin_summe_0_20260909_163954-tag-Train_step_pi_ratio.csv",
    'MiniCPM DPO + Margin': root + "run-vslice_DPO_summe_0_20260905_104913-tag-Train_step_pi_ratio.csv",
    # 'MiniCPM Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_pi_ratio.csv",
    'QWEN2-VL': root + "run-vslice_qwen2_vl_3b_DPO_no_margin_summe_0_20260921_181616-tag-Train_step_pi_ratio.csv",
    'QWEN2-VL DPO + Margin': root + "run-vslice_qwen2_vl_3b_DPO_summe_0_20260917_053526-tag-Train_step_pi_ratio.csv"
}

plt.rcParams.update({'font.size': 18})

# Basic simple colors: Blue for MiniCPM, Red for QWEN2-VL to contrast margin effects
_COLORS = ['tab:orange', 'tab:blue', 'tab:green', 'tab:red']
_STYLES = ['--', '-', '--', '-']
_LINEWIDTHS = [1.0, 1.2, 1.0, 1.2]

def _get_curve_style(label, idx):
    """Determine linestyle, linewidth, and color to contrast normal DPO vs DPO + Margin."""
    is_margin = 'margin' in label.lower()
    style = '-' if is_margin else '-.'
    lw = 2 if is_margin else 1.5
    color = _COLORS[idx % len(_COLORS)]
    return style, lw, color

def _plot_metric_on_ax(ax, file_paths, ylabel, smoothing_weight, start,
                       legend_loc='upper right', show_legend=False):
    """Helper: draw one metric panel onto an existing Axes."""
    for idx, (label, file_path) in enumerate(file_paths.items()):
        df = pd.read_csv(file_path)
        style, lw, color = _get_curve_style(label, idx)
        if start is not None:
            ax.plot(df[start:-1]['Step'],
                    moving_average(df[start:-1]['Value'], smoothing_weight, start=0),
                    style,
                    color=color,
                    label=label, linewidth=lw)
        else:
            ax.plot(df['Step'],
                    moving_average(df['Value'], smoothing_weight),
                    style,
                    color=color,
                    label=label, linewidth=lw)
    ax.set_xlabel('Steps')
    ax.set_ylabel(ylabel)
    ax.set_xlim(0,575)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    if show_legend:
        ax.legend(loc=legend_loc, prop={'size': 13})
    ax.grid(True, linestyle='--', alpha=0.6)

def _plot_kl_vs_margin_on_ax(ax, kl_paths, margin_paths, smoothing_weight, start, every=5,
                             text_offsets=None, n_end=10):
    """Helper: scatter smoothed separation margin (x) against KL divergence (y), aligned by step,
    and enclose each run in a convex hull to contrast DPO with and without margin.
    For each model, an arrow joins the final state of the no-margin run to the margin run,
    labelled with the change in KL spent per unit separation (lower is better)."""
    # Label placement per model in axes fraction (x, y, ha, va); runs lie on the diagonal,
    # so the upper-left and lower-right corners are empty
    text_offsets = text_offsets or {'MiniCPM': (0.03, 0.97, 'left', 'top'),
                                    'QWEN2-VL': (0.97, 0.03, 'right', 'bottom')}
    ends = {}  # model -> {is_margin: (end_margin, end_kl, color)}
    for idx, label in enumerate(kl_paths):
        kl = pd.read_csv(kl_paths[label])[['Step', 'Value']]
        mg = pd.read_csv(margin_paths[label])[['Step', 'Value']]
        df = kl.merge(mg, on='Step', suffixes=('_kl', '_margin'))[start:-1]
        style, lw, color = _get_curve_style(label, idx)
        full = np.column_stack([moving_average(df['Value_margin'], smoothing_weight),
                                moving_average(df['Value_kl'], smoothing_weight)])
        pts = full[::every]
        is_margin = 'margin' in label.lower()
        end_mg, end_kl = full[-n_end:].mean(axis=0)
        ends.setdefault(label.split()[0], {})[is_margin] = (end_mg, end_kl, color)
        ax.scatter(pts[:, 0], pts[:, 1], s=12, color=color, alpha=0.6,
                   marker='o' if is_margin else 'x', label=label)
        if len(pts) >= 3:
            hull = pts[ConvexHull(pts).vertices]
            ax.fill(hull[:, 0], hull[:, 1], color=color, alpha=0.12)
            ax.plot(np.append(hull[:, 0], hull[0, 0]), np.append(hull[:, 1], hull[0, 1]),
                    style, color=color, linewidth=lw)

    for model, runs in ends.items():
        if True not in runs or False not in runs:
            continue
        (mg_n, kl_n, _), (mg_m, kl_m, color_m) = runs[False], runs[True]
        # End-state markers: hollow circle = no margin, yellow star = margin; arrow drawn above the hulls
        ax.scatter([mg_n], [kl_n], s=70, facecolors='white', edgecolors='black', linewidths=1.5, zorder=7)
        ax.scatter([mg_m], [kl_m], s=260, color='gold', edgecolors='black', linewidths=1.2,
                   marker='*', zorder=8)
        ax.annotate('', xy=(mg_m, kl_m), xytext=(mg_n, kl_n), zorder=7,
                    arrowprops=dict(arrowstyle='-|>', color='black', lw=1.8, mutation_scale=16,
                                    shrinkA=4, shrinkB=9))
        tx, ty, ha, va = text_offsets.get(model, (0.03, 0.97, 'left', 'top'))
        ax.text(tx, ty, f'{model}: +Margin\nKL/Sep {kl_n / mg_n:.2f}' + r'$\rightarrow$' + f'{kl_m / mg_m:.2f}',
                transform=ax.transAxes, ha=ha, va=va, fontsize=10, color=color_m, zorder=9,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=color_m, lw=0.8, alpha=0.9))
    ax.set_xlabel('Separation Margin')
    ax.set_ylabel('KL Divergence')
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_combined(save_file="vslice_combined.pdf"):
    """Compile Loss, KL Divergence, Margin and KL vs. Margin into a 1×4 figure."""
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))

    configs = [
        (file_paths_loss,     'Loss',           0.96, 7,  'upper right', False),
        (file_paths_kl,       'KL Divergence', 0.96, 7,  'lower right', False),
        (file_paths_pi_ratio, 'Separation Margin',         0.96, 7,  'upper left', True),
    ]

    for ax, (fps, ylabel, sw, st, lloc, show_leg) in zip(axes, configs):
        _plot_metric_on_ax(ax, fps, ylabel, sw, st, legend_loc=lloc, show_legend=show_leg)

    _plot_kl_vs_margin_on_ax(axes[3], file_paths_kl, file_paths_pi_ratio, 0.96, 7)

    plt.tight_layout()
    plt.savefig("./results/" + save_file, format="pdf", bbox_inches="tight")
    plt.show()

def plot_loss(file_paths, loss_type, save_file, smoothing_weight=0.99, start=7):
    """Plots the loss over steps for given file paths."""
    plt.figure(figsize=(5,4))
    for idx, (label, file_path) in enumerate(file_paths.items()):
        df = pd.read_csv(file_path)
        style, lw, color = _get_curve_style(label, idx)
        
        if start is not None:
            plt.plot(df[start:-1]['Step'], moving_average(df[start: -1]['Value'], smoothing_weight, start=0), style, color=color, linewidth=lw, label=label)
        else:
            plt.plot(df['Step'], moving_average(df['Value'], smoothing_weight), style, color=color, linewidth=lw, label=label)

    plt.xlabel('Steps')
    plt.ylabel(loss_type)
    #plt.title('Train Loss over Steps')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("./results/" + save_file, format="pdf", bbox_inches="tight")
    plt.show()

plot_combined(save_file="vslice_combined.pdf")