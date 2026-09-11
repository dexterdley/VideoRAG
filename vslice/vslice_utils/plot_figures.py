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
    'DPO': root + "run-vslice_DPO_no_margin_summe_0_20260909_163954-tag-Train_step_kl_drift.csv",
    'DPO + Margin': root + "run-vslice_DPO_summe_0_20260905_104913-tag-Train_step_kl_drift.csv",
    'Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_kl_drift.csv",
}

file_paths_loss = {
    'DPO': root + "run-vslice_DPO_no_margin_summe_0_20260909_163954-tag-Train_step_loss.csv",
    'DPO + Margin': root + "run-vslice_DPO_summe_0_20260905_104913-tag-Train_step_loss.csv",
    'Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_loss.csv",
}

file_paths_pi_ratio = {
    'DPO': root + "run-vslice_DPO_no_margin_summe_0_20260909_163954-tag-Train_step_pi_ratio.csv",
    'DPO + Margin': root + "run-vslice_DPO_summe_0_20260905_104913-tag-Train_step_pi_ratio.csv",
    'Focal DPO + Margin': root + "run-vslice_MPO_summe_0_20260904_230053-tag-Train_step_pi_ratio.csv",
}

plt.rcParams.update({'font.size': 18})

# Curated colour palette – distinct, publication-friendly
_COLORS = ['#2E86AB', '#E84855', 'g']   # steel-blue, crimson, emerald
_STYLES = ['-', '-', '-']

def _plot_metric_on_ax(ax, file_paths, ylabel, smoothing_weight, start,
                       legend_loc='upper right', show_legend=False):
    """Helper: draw one metric panel onto an existing Axes."""
    for idx, (label, file_path) in enumerate(file_paths.items()):
        df = pd.read_csv(file_path)
        if start is not None:
            ax.plot(df[start:-1]['Step'],
                    moving_average(df[start:-1]['Value'], smoothing_weight, start=0),
                    _STYLES[idx % len(_STYLES)],
                    color=_COLORS[idx % len(_COLORS)],
                    label=label, linewidth=2)
        else:
            ax.plot(df['Step'],
                    moving_average(df['Value'], smoothing_weight),
                    _STYLES[idx % len(_STYLES)],
                    color=_COLORS[idx % len(_COLORS)],
                    label=label, linewidth=2)
    ax.set_xlabel('Steps')
    ax.set_ylabel(ylabel)
    if show_legend:
        ax.legend(loc=legend_loc, prop={'size': 13})
    ax.grid(True, linestyle='--', alpha=0.6)

def plot_combined(save_file="vslice_combined.pdf"):
    """Compile KL Divergence, Loss and Margin into a 1×3 figure."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    configs = [
        (file_paths_loss,     'Loss',           0.9, 7,  'upper right', True),
        (file_paths_kl,       'KL Divergence', 0.9, 7,  'lower right', False),
        (file_paths_pi_ratio, 'Separation Margin',         0.9, 7,  'lower right', False),
    ]

    for ax, (fps, ylabel, sw, st, lloc, show_leg) in zip(axes, configs):
        _plot_metric_on_ax(ax, fps, ylabel, sw, st, legend_loc=lloc, show_legend=show_leg)

    plt.tight_layout()
    plt.savefig("./results/" + save_file, format="pdf", bbox_inches="tight")
    plt.show()

def plot_loss(file_paths, loss_type, save_file, smoothing_weight=0.99, start=7):
    """Plots the loss over steps for given file paths."""
    plt.figure(figsize=(5,4))
    for label, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        
        if start is not None:
            plt.plot(df[start:-1]['Step'], moving_average(df[start: -1]['Value'], smoothing_weight, start=0), '-' ,label=label)
        else:
            plt.plot(df['Step'], moving_average(df['Value'], smoothing_weight), label=label)

    plt.xlabel('Steps')
    plt.ylabel(loss_type)
    #plt.title('Train Loss over Steps')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig("./results/" + save_file, format="pdf", bbox_inches="tight")
    plt.show()
    
def plot_margin(file_paths, margin_file_path, loss_type, save_file, smoothing_weight=0.99, start=10):
    """Plots the loss over steps for given file paths."""
    plt.figure(figsize=(5,4))
    margin_df = pd.read_csv(margin_file_path)
    for label, file_path in file_paths.items():
        df = pd.read_csv(file_path)
        if start is not None:
            plt.plot(df[start:-1]['Step'], moving_average(df[start: -1]['Value']/margin_df[start: -1]['Value'], smoothing_weight, start=0), '-' ,label=label)
        else:
            plt.plot(df['Step'], moving_average(df['Value']/margin_df['Value'], smoothing_weight), label=label)

    plt.xlabel('Steps')
    plt.ylabel(loss_type)
    #plt.title('Train Loss over Steps')
    plt.legend()
    plt.grid(True)
    plt.savefig("./results/" + save_file, format="pdf", bbox_inches="tight")
    if loss_type == "VORD Loss":
        plt.ylim(0.15, 0.55)
    plt.show()

def draw_cluster(axs, x, y, c='cyan'):
    points = np.stack([x, y ]).T
    hull = ConvexHull(points)
    x_hull = np.append(points[hull.vertices,0],
                       points[hull.vertices,0][0])
    y_hull = np.append(points[hull.vertices,1],
                       points[hull.vertices,1][0])
    
    dist = np.sqrt((x_hull[:-1] - x_hull[1:])**2 + (y_hull[:-1] - y_hull[1:])**2)
    dist_along = np.concatenate(([0], dist.cumsum()))
    #spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along, s=0, per=1)
    spline, u = interpolate.splprep([x_hull, y_hull], u=dist_along)
    interp_d = np.linspace(dist_along[0], dist_along[-1], 50)
    interp_x, interp_y = interpolate.splev(interp_d, spline)
    
    axs.fill(interp_x, interp_y, alpha=0.1, color=c)

def plot_scatter_patterns(x_file_paths, y_file_paths, x_label, y_label, save_file, smoothing_weight=0.9, start=0):
    """
    Plots scatter patterns between two metrics over steps.
    """
    plt.figure(figsize=(5, 4))
    ax = plt.gca()
    
    colors = ['b', 'y', 'g']
    
    for i, key in enumerate(x_file_paths):
        x_file_path = x_file_paths[key]
        y_file_path = y_file_paths[key]

        x_df = pd.read_csv(x_file_path)
        y_df = pd.read_csv(y_file_path)
        
        # Apply smoothing
        smoothed_x = moving_average(x_df[start:-1]['Value'], smoothing_weight, start=start)
        smoothed_y = moving_average(y_df[start:-1]['Value'], smoothing_weight, start=start)

        # Note: The original scatter plot had `3000 - smoothed_x`.
        # Assuming this is a desired transformation, apply it here.
        # If not, use `smoothed_x` directly.
        plt.scatter(3000 - smoothed_x, smoothed_y, label=key, alpha=0.8)
        draw_cluster(ax, 3000 - smoothed_x, smoothed_y, c=colors[i])
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)

    plt.savefig("./results/" + save_file, format="pdf", bbox_inches="tight")
    plt.show()


plot_combined(save_file="vslice_combined.pdf")

#plot_margin(file_paths_ordinal_ent, margin_file_path, 'KL Divergence/mθ', save_file="paligemma_ordinal_ent_margin.pdf", smoothing_weight=0.98, start=30)
#plot_scatter_patterns(file_paths_violations, file_paths_SNR, "Violations reduced", "Signal-to-Noise ratio", save_file="paligemma_snr_violations.pdf", smoothing_weight=0.9, start=35)