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
    'QWEN2-VL': root + "run-vslice_qwen2_vl_3b_DPO_no_margin_summe_0_20260919_042200-tag-Train_step_kl_drift.csv",
    'QWEN2-VL DPO + Margin': root + "run-vslice_qwen2_vl_3b_DPO_summe_0_20260917_053526-tag-Train_step_kl_drift.csv"
     # 'Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_kl_drift.csv",
}

file_paths_loss = {
    'MiniCPM DPO': root + "run-vslice_DPO_no_margin_summe_0_20260909_163954-tag-Train_step_loss.csv",
    'MiniCPM DPO + Margin': root + "run-vslice_DPO_summe_0_20260905_104913-tag-Train_step_loss.csv",
    # 'QWEN2-VL': root + "run-vslice_qwen2_vl_3b_DPO_no_margin_summe_0_20260919_042200-tag-Train_step_loss.csv",
    'QWEN2-VL DPO + Margin': root + "run-vslice_qwen2_vl_3b_DPO_summe_0_20260917_053526-tag-Train_step_loss.csv",
    # 'Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_loss.csv",
}

file_paths_pi_ratio = {
    'MiniCPM DPO': root + "run-vslice_DPO_no_margin_summe_0_20260909_163954-tag-Train_step_pi_ratio.csv",
    'MiniCPM DPO + Margin': root + "run-vslice_DPO_summe_0_20260905_104913-tag-Train_step_pi_ratio.csv",
    'QWEN2-VL': root + "run-vslice_qwen2_vl_3b_DPO_no_margin_summe_0_20260919_042200-tag-Train_step_pi_ratio.csv",
    'QWEN2-VL DPO + Margin': root + "run-vslice_qwen2_vl_3b_DPO_summe_0_20260917_053526-tag-Train_step_pi_ratio.csv",
    # 'Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_pi_ratio.csv",
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
    if show_legend:
        ax.legend(loc=legend_loc, prop={'size': 13})
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_combined(save_file="vslice_combined.pdf"):
    """Compile KL Divergence, Loss and Margin into a 1×3 figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    configs = [
        (file_paths_loss,     'Loss',           0.9, 7,  'upper right', False),
        (file_paths_kl,       'KL Divergence', 0.9, 7,  'lower right', False),
        (file_paths_pi_ratio, 'Separation Margin',         0.9, 7,  'upper left', True),
    ]

    for ax, (fps, ylabel, sw, st, lloc, show_leg) in zip(axes, configs):
        _plot_metric_on_ax(ax, fps, ylabel, sw, st, legend_loc=lloc, show_legend=show_leg)

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