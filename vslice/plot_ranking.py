import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==============================================================================
# Data Loading & Global Aggregation
# ==============================================================================
results_dir = "./results" if os.path.exists("./results") else "../results"
base_csv_paths = [os.path.join(results_dir, f"zero_shot_ranking_split{i}.csv") for i in range(5)]
peft_csv_paths = [os.path.join(results_dir, f"zero_shot_ranking_minicpm_summe_20260905_104913_best_DPO_split_split_{i}.csv") for i in range(5)]

df_base = pd.concat([pd.read_csv(path) for path in base_csv_paths], ignore_index=True)
df_peft = pd.concat([pd.read_csv(path) for path in peft_csv_paths], ignore_index=True)
print(f"\nAll 5 CSVs loaded! Concatenated into global DataFrame (Base: {len(df_base):,}, PEFT: {len(df_peft):,})...")
os.makedirs(results_dir, exist_ok=True)

# ==============================================================================
# Typography & Figure Dimensions for CVPR Double-Column Format
# ==============================================================================
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
    'font.size': 18,
    'axes.labelsize': 12.0,
    'axes.titlesize': 10,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 5,
    'lines.markersize': 5.5,
    'grid.alpha': 0.5,
    'grid.linestyle': '--',
})

FIG_SIZE = (5, 4)


# ==============================================================================
# Figure 1: Coarse Recognition vs. Fine-Grained Ranking Discrepancy (Tiers)
# Proves: Model can recognize coarse differences, but fails at fine-grained ranking.
#         Alignment elevates fine-grained accuracy from 54.6% -> 75.0%.
# ==============================================================================
fig1, ax1 = plt.subplots(figsize=FIG_SIZE)

tiers = [
    (r"Fine" + "\n" + r"($\Delta z < 0.25$)", df_base['human_gap'] < 0.25),
    (r"Moderate" + "\n" + r"($0.25 \leq \Delta z < 0.5$)", (df_base['human_gap'] >= 0.25) & (df_base['human_gap'] < 0.5)),
    (r"Coarse" + "\n" + r"($\Delta z \geq 0.5$)", df_base['human_gap'] >= 0.5),
]

x = np.arange(len(tiers))
width = 0.35

base_accs = [df_base[mask]['correct_ranking'].mean() * 100 for _, mask in tiers]
peft_accs = [df_peft[mask]['correct_ranking'].mean() * 100 for _, mask in tiers]

bars1 = ax1.bar(x - width/2, base_accs, width, label='Zero-Shot Base', color='#1f77b4', edgecolor='black', linewidth=0.6, alpha=0.85)
bars2 = ax1.bar(x + width/2, peft_accs, width, label='PEFT Aligned (Ours)', color='#2ca02c', edgecolor='black', linewidth=0.6, alpha=0.85)

ax1.axhline(50, color='gray', linestyle='--', linewidth=1.0, label='Chance (50%)')
ax1.set_xticks(x)
ax1.set_xticklabels([name for name, _ in tiers], weight='bold')
ax1.set_ylabel('Pairwise Accuracy (%)', weight='bold')
ax1.set_ylim(40, 95)
ax1.set_title('Recognition vs. Ranking Gap', weight='bold', pad=8)
ax1.legend(loc='upper left', frameon=True, fontsize=7.5, ncol=1)
ax1.grid(True, axis='y')

for bar in bars1:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 1.0, f'{y:.1f}%', ha='center', va='bottom', fontsize=10.5, color='#1f77b4', weight='bold')
for bar in bars2:
    y = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2.0, y + 1.0, f'{y:.1f}%', ha='center', va='bottom', fontsize=10.5, color='#2ca02c', weight='bold')

plt.tight_layout()
out_f1_png = os.path.join(results_dir, 'cvpr_fig1_recognition_vs_ranking_tiers.png')
out_f1_pdf = os.path.join(results_dir, 'cvpr_fig1_recognition_vs_ranking_tiers.pdf')
plt.savefig(out_f1_png, dpi=300, bbox_inches='tight')
plt.savefig(out_f1_pdf, dpi=300, bbox_inches='tight')
plt.close(fig1)
print(f'Saved Figure 1 to {out_f1_png} and {out_f1_pdf}')


# ==============================================================================
# Figure 2: Confidence-Accuracy Calibration Curve
# Proves: Zero-shot confidence does not imply ranking correctness (inverts at low confidence).
#         PEFT alignment achieves monotonic confidence calibration (68.6% -> 83.9%).
# ==============================================================================
fig2, ax2 = plt.subplots(figsize=FIG_SIZE)

conf_labels = ['Q1\n(Low)', 'Q2', 'Q3', 'Q4', 'Q5\n(High)']
df_base['conf_q'] = pd.qcut(df_base['model_gap'].abs(), q=5, labels=conf_labels)
df_peft['conf_q'] = pd.qcut(df_peft['model_gap'].abs(), q=5, labels=conf_labels)

cal_base = df_base.groupby('conf_q', observed=False)['correct_ranking'].mean() * 100
cal_peft = df_peft.groupby('conf_q', observed=False)['correct_ranking'].mean() * 100

qx = np.arange(len(conf_labels))
ax2.plot(qx, cal_base.values, marker='o', markersize=5.5, linestyle='-.', lw=1.8, color='#1f77b4', label='Zero-Shot Base', zorder=3)
ax2.plot(qx, cal_peft.values, marker='s', markersize=5.5, linestyle='-', lw=2.2, color='#2ca02c', label='VSLICE (Ours)', zorder=3)
ax2.plot([0, 4], [50, 100], color='#555555', linestyle=':', linewidth=1.5, label='Ideal Calibration', zorder=2)
ax2.axhline(50, color='gray', linestyle='--', linewidth=1.0, label='Random (50%)', zorder=1)

for i, (b_val, p_val) in enumerate(zip(cal_base.values, cal_peft.values)):
    ax2.text(i, b_val - 4.5, f'{b_val:.1f}%', ha='center', fontsize=10.5, color='#1f77b4', weight='bold')
    ax2.text(i, p_val + 2.0, f'{p_val:.1f}%', ha='center', fontsize=10.5, color='#2ca02c', weight='bold')

# <-> arrow showing alignment improvement
diff_q0 = cal_peft.values[0] - cal_base.values[0]
ax2.annotate('', xy=(0, cal_peft.values[0] - 1.8), xytext=(0, cal_base.values[0] + 1.8),
             arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2.0, shrinkA=0, shrinkB=0))
ax2.text(0.08, (cal_peft.values[0] + cal_base.values[0]) / 2 + 8, f'+{diff_q0:.1f}%',
         ha='left', va='center', fontsize=10.5, color='#d62728', weight='bold',
         bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

ax2.set_xticks(qx)
ax2.set_xticklabels(conf_labels, weight='bold')
ax2.set_ylim(18, 104)
ax2.set_ylabel('Pairwise Accuracy (%)', weight='bold')
ax2.set_xlabel(r'Confidence Quantile ($|\Delta \hat{z}|$)', weight='bold')
ax2.set_title('Confidence Ranking', weight='bold', pad=8)
ax2.legend(loc='lower right', frameon=True, fontsize=8)
ax2.grid(True)

plt.tight_layout()
out_f2_png = os.path.join(results_dir, 'cvpr_fig2_confidence_ranking.png')
out_f2_pdf = os.path.join(results_dir, 'cvpr_fig2_confidence_ranking.pdf')
plt.savefig(out_f2_png, dpi=300, bbox_inches='tight')
plt.savefig(out_f2_pdf, dpi=300, bbox_inches='tight')
plt.close(fig2)
print(f'Saved Figure 2 to {out_f2_png} and {out_f2_pdf}')


# ==============================================================================
# Figure 3: Ranking State Transition & Error Rescue Flow
# Proves: Alignment rescues 57.7% of Zero-Shot errors while retaining 93.3% of correct pairs.
# ==============================================================================
fig3, ax3 = plt.subplots(figsize=FIG_SIZE)

base_wrong = (df_base['correct_ranking'] == 0)
base_right = (df_base['correct_ranking'] == 1)

n_wrong = base_wrong.sum()
n_right = base_right.sum()

rescued = (base_wrong & (df_peft['correct_ranking'] == 1)).sum()
unresolved = (base_wrong & (df_peft['correct_ranking'] == 0)).sum()
maintained = (base_right & (df_peft['correct_ranking'] == 1)).sum()
regressed = (base_right & (df_peft['correct_ranking'] == 0)).sum()

pct_rescued = (rescued / n_wrong) * 100
pct_unres = (unresolved / n_wrong) * 100
pct_maint = (maintained / n_right) * 100
pct_reg = (regressed / n_right) * 100

groups = [f'Zero-Shot Errors\n(n={n_wrong:,})', f'Zero-Shot Correct\n(n={n_right:,})']
cx = np.arange(len(groups))
cwidth = 0.44

ax3.bar(cx[0], pct_rescued, cwidth, label='Rescued', color='#2ca02c', edgecolor='black', linewidth=0.6, alpha=0.85)
ax3.bar(cx[0], pct_unres, cwidth, bottom=pct_rescued, label='Unresolved', color='#e0e0e0', edgecolor='black', linewidth=0.6)

ax3.bar(cx[1], pct_maint, cwidth, label='Retained', color='#1f77b4', edgecolor='black', linewidth=0.6, alpha=0.85)
ax3.bar(cx[1], pct_reg, cwidth, bottom=pct_maint, label='Regressed', color='#d62728', edgecolor='black', linewidth=0.6, alpha=0.7)

ax3.text(cx[0], pct_rescued / 2, f'Rescued\n{pct_rescued:.1f}%\n({rescued:,})', ha='center', va='center', fontsize=10.5, color='white', weight='bold')
ax3.text(cx[0], pct_rescued + pct_unres / 2, f'Unresolved\n{pct_unres:.1f}%\n({unresolved:,})', ha='center', va='center', fontsize=10.5, color='black', weight='bold')

ax3.text(cx[1], pct_maint / 2, f'Retained\n{pct_maint:.1f}%\n({maintained:,})', ha='center', va='center', fontsize=10.5, color='white', weight='bold')
ax3.text(cx[1], pct_maint + pct_reg / 2, f'Regressed\n{pct_reg:.1f}%', ha='center', va='center', fontsize=10.5, color='white', weight='bold')

net_gain = rescued - regressed
ax3.text(0.5, 105, f'Net Gain: +{net_gain:,} pairs (+22.0%)',
         ha='center', va='bottom', fontsize=8.0, weight='bold',
         bbox=dict(boxstyle='round,pad=0.25', facecolor='#e8f5e9', edgecolor='#2ca02c', linewidth=0.8))

ax3.set_xticks(cx)
ax3.set_xticklabels(groups, weight='bold')
ax3.set_ylabel('Proportion (%)', weight='bold')
ax3.set_ylim(0, 120)
ax3.set_title('Ranking Error Rescue Flow', weight='bold', pad=8)
ax3.legend(loc='lower center', ncol=2, frameon=True, fontsize=8)
ax3.grid(True, axis='y')

plt.tight_layout()
out_f3_png = os.path.join(results_dir, 'cvpr_fig3_ranking_state_rescue.png')
out_f3_pdf = os.path.join(results_dir, 'cvpr_fig3_ranking_state_rescue.pdf')
plt.savefig(out_f3_png, dpi=300, bbox_inches='tight')
plt.savefig(out_f3_pdf, dpi=300, bbox_inches='tight')
plt.close(fig3)
print(f'Saved Figure 3 to {out_f3_png} and {out_f3_pdf}')


# ==============================================================================
# Figure 4: Continuous Difficulty Curve (Accuracy vs. True Human Margin)
# Proves: Across the full continuous difficulty spectrum, PEFT strictly dominates Zero-Shot.
# ==============================================================================
fig4, ax4 = plt.subplots(figsize=FIG_SIZE)

df_base['difficulty_bin'] = pd.qcut(df_base['human_gap'], q=5, duplicates='drop')
bins = df_base['difficulty_bin'].cat.categories
df_peft['difficulty_bin'] = pd.cut(df_peft['human_gap'], bins=bins, include_lowest=True)

stats_base = df_base.groupby('difficulty_bin', observed=False).agg(
    accuracy=('correct_ranking', 'mean'),
    mean_human_gap=('human_gap', 'mean')
).reset_index()
stats_base['accuracy'] *= 100

stats_peft = df_peft.groupby('difficulty_bin', observed=False).agg(
    accuracy=('correct_ranking', 'mean'),
    mean_human_gap=('human_gap', 'mean')
).reset_index()
stats_peft['accuracy'] *= 100

ax4.plot(stats_base['mean_human_gap'], stats_base['accuracy'], 
         marker='o', markersize=5.5, linestyle='-.', linewidth=1.8, 
         color='#1f77b4', label='Zero-Shot Base')
ax4.plot(stats_peft['mean_human_gap'], stats_peft['accuracy'], 
         marker='s', markersize=5.5, linestyle='-', linewidth=2.2, 
         color='#2ca02c', label='VSLICE (Ours)')

ax4.axhline(50, color='gray', linestyle='--', linewidth=1.0, label='Chance (50%)')

for i, row in stats_base.iterrows():
    ax4.text(row['mean_human_gap'], row['accuracy'] - 3.2, f"{row['accuracy']:.1f}", ha='center', fontsize=10.5, color='#1f77b4', weight='bold')

for i, row in stats_peft.iterrows():
    ax4.text(row['mean_human_gap'], row['accuracy'] + 1.8, f"{row['accuracy']:.1f}", ha='center', fontsize=10.5, color='#2ca02c', weight='bold')

# <-> arrow showing alignment improvement
x_arr = stats_base.iloc[2]['mean_human_gap']
y_base = stats_base.iloc[2]['accuracy']
y_peft = stats_peft.iloc[2]['accuracy']
diff_pt = y_peft - y_base
ax4.annotate('', xy=(x_arr, y_peft - 1.8), xytext=(x_arr, y_base + 1.8),
             arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2.0, shrinkA=0, shrinkB=0))
ax4.text(x_arr + 0.012, (y_peft + y_base) / 2, f'+{diff_pt:.1f}%',
         ha='left', va='center', fontsize=10.5, color='#d62728', weight='bold')

ax4.set_title('Preference Accuracy vs. Highlight Difficulty', weight='bold', pad=8)
ax4.set_xlabel(r'True Human Gap ($\Delta z$)', weight='bold')
ax4.set_ylabel('Preference Accuracy (%)', weight='bold')
ax4.set_xlim(0.18, 0.72)
ax4.set_ylim(40, 95)
ax4.legend(loc='lower right', frameon=True, fontsize=8)
ax4.grid(True)

plt.tight_layout()
out_f4_png = os.path.join(results_dir, 'cvpr_fig4_accuracy_vs_difficulty.png')
out_f4_pdf = os.path.join(results_dir, 'cvpr_fig4_accuracy_vs_difficulty.pdf')
plt.savefig(out_f4_png, dpi=300, bbox_inches='tight')
plt.savefig(out_f4_pdf, dpi=300, bbox_inches='tight')
plt.close(fig4)
print(f'Saved Figure 4 to {out_f4_png} and {out_f4_pdf}')


# ==============================================================================
# Combined 3-Subplot Figure: Fig 1, Fig 2, and Fig 4
# Maintained: Aspect ratio (5:4 per subplot -> 15.5:4.2 overall), fontsizes, texts, arrows
# ==============================================================================
fig_comb, (cax1, cax2, cax3) = plt.subplots(1, 3, figsize=(13.2, 3.9))
font_sz = 12.5
# ------------------------------------------------------------------------------
# Subplot 1: Recognition vs. Ranking Gap (Tiers)
# ------------------------------------------------------------------------------
bars1 = cax1.bar(x - width/2, base_accs, width, label='Zero-Shot Base', color='#1f77b4', edgecolor='black', linewidth=0.6, alpha=0.85)
bars2 = cax1.bar(x + width/2, peft_accs, width, label='VSLICE (Ours)', color='#2ca02c', edgecolor='black', linewidth=0.6, alpha=0.85)

cax1.axhline(50, color='gray', linestyle='--', linewidth=1.0, label='Random (50%)')
cax1.set_xticks(x)
cax1.set_xticklabels([name for name, _ in tiers], weight='bold')
cax1.set_ylabel('Pairwise Accuracy (%)', weight='bold')
cax1.set_ylim(40, 95)
cax1.set_title('Recognition vs. Ranking Gap', weight='bold', fontsize=font_sz, pad=8)
cax1.legend(loc='upper left', frameon=True, fontsize=8, ncol=1)
cax1.grid(True, axis='y')

for bar in bars1:
    y = bar.get_height()
    cax1.text(bar.get_x() + bar.get_width()/2.0, y + 1.0, f'{y:.1f}%', ha='center', va='bottom', fontsize=10.5, color='#1f77b4', weight='bold')
for bar in bars2:
    y = bar.get_height()
    cax1.text(bar.get_x() + bar.get_width()/2.0, y + 1.0, f'{y:.1f}%', ha='center', va='bottom', fontsize=10.5, color='#2ca02c', weight='bold')

# ------------------------------------------------------------------------------
# Subplot 2: Preference Accuracy vs. Highlight Difficulty (with <-> arrow)
# ------------------------------------------------------------------------------
cax2.plot(stats_base['mean_human_gap'], stats_base['accuracy'], 
          marker='o', markersize=5.5, linestyle='-.', linewidth=1.8, 
          color='#1f77b4', label='Zero-Shot Base')
cax2.plot(stats_peft['mean_human_gap'], stats_peft['accuracy'], 
          marker='s', markersize=5.5, linestyle='-', linewidth=2.2, 
          color='#2ca02c', label='VSLICE (Ours)')
cax2.axhline(50, color='gray', linestyle='--', linewidth=1.0, label='Random (50%)')

for i, row in stats_base.iterrows():
    cax2.text(row['mean_human_gap'], row['accuracy'] - 3.2, f"{row['accuracy']:.1f}", ha='center', fontsize=11.5, color='#1f77b4', weight='bold')
for i, row in stats_peft.iterrows():
    cax2.text(row['mean_human_gap'], row['accuracy'] + 1.8, f"{row['accuracy']:.1f}", ha='center', fontsize=11.5, color='#2ca02c', weight='bold')

cax2.annotate('', xy=(x_arr, y_peft - 1.8), xytext=(x_arr, y_base + 1.8),
              arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2.0, shrinkA=0, shrinkB=0))
cax2.text(x_arr + 0.012, (y_peft + y_base) / 2, f'+{diff_pt:.1f}%',
          ha='left', va='center', fontsize=font_sz, color='#d62728', weight='bold')

cax2.set_title('Preference Accuracy vs. Highlight Difficulty', weight='bold', fontsize=font_sz, pad=8)
cax2.set_xlabel(r'True Human Gap ($\Delta z$)', weight='bold')
cax2.set_ylabel('Preference Accuracy (%)', weight='bold')
cax2.set_xlim(0.18, 0.72)
cax2.set_ylim(40, 95)
cax2.legend(loc='lower right', frameon=True, fontsize=8)
cax2.grid(True)


# ------------------------------------------------------------------------------
# Subplot 3: Confidence Calibration Curve (with <-> arrow)
# ------------------------------------------------------------------------------
cax3.plot(qx, cal_base.values, marker='o', markersize=5.5, linestyle='-.', lw=1.8, color='#1f77b4', label='Zero-Shot Base', zorder=3)
cax3.plot(qx, cal_peft.values, marker='s', markersize=5.5, linestyle='-', lw=2.2, color='#2ca02c', label='VSLICE (Ours)', zorder=3)
cax3.plot([0, 4], [50, 100], color='#555555', linestyle=':', linewidth=1.5, label='Ideal Calibration', zorder=2)
cax3.axhline(50, color='gray', linestyle='--', linewidth=1.0, label='Random (50%)', zorder=1)

for i, (b_val, p_val) in enumerate(zip(cal_base.values, cal_peft.values)):
    cax3.text(i, b_val - 4.5, f'{b_val:.1f}%', ha='center', fontsize=11.5, color='#1f77b4', weight='bold')
    cax3.text(i, p_val + 2.0, f'{p_val:.1f}%', ha='center', fontsize=11.5, color='#2ca02c', weight='bold')

cax3.annotate('', xy=(0, cal_peft.values[0] - 1.8), xytext=(0, cal_base.values[0] + 1.8),
              arrowprops=dict(arrowstyle='<->', color='#d62728', lw=2.0, shrinkA=0, shrinkB=0))
cax3.text(0.08, (cal_peft.values[0] + cal_base.values[0]) / 2 + 8, f'+{diff_q0:.1f}%',
          ha='left', va='center', fontsize=font_sz, color='#d62728', weight='bold',
          bbox=dict(boxstyle='round,pad=0.15', facecolor='white', edgecolor='none', alpha=0.85))

cax3.set_xticks(qx)
cax3.set_xticklabels(conf_labels)
cax3.set_ylim(18, 104)
cax3.set_ylabel('Pairwise Accuracy (%)', weight='bold')
cax3.set_xlabel(r'Confidence Quantile ($|\Delta \hat{z}|$)', weight='bold')
cax3.set_title('Confidence Ranking', weight='bold', fontsize=font_sz, pad=8)
cax3.legend(loc='lower right', frameon=True, fontsize=8)
cax3.grid(True)

plt.tight_layout(pad=0.4, w_pad=0.4)
out_comb_png = os.path.join(results_dir, 'cvpr_combined_3subplot.png')
out_comb_pdf = os.path.join(results_dir, 'cvpr_combined_3subplot.pdf')
plt.savefig(out_comb_png, dpi=300, bbox_inches='tight')
plt.savefig(out_comb_pdf, dpi=300, bbox_inches='tight')
plt.close(fig_comb)
print(f'Saved Combined 3-Subplot Figure to {out_comb_png} and {out_comb_pdf}')

print('\nAll figures successfully generated and saved!')
