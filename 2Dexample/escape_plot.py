#!/usr/bin/env python3
"""Plot escape benchmark results from LaTeX table data."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Data from LaTeX table ──
methods = ["Co-SDF Grad", "Co-SDF Samp", "SIREN Grad", "SIREN Samp", "MLP Grad", "MLP Samp"]
worlds  = ["2 Obs", "4 Obs", "6 Obs"]

#                       2 obs    4 obs    6 obs
success = np.array([
    [100.0, 97.9, 100.0],
    [100.0, 100.0, 100.0],
    [20.4, 18.5, 23.8],
    [22.4, 20.9, 24.8],
    [23.7, 77.8, 68.2],
    [28.7, 79.2, 68.4],
])
steps = np.array([
    [20.8, 27.8, 23.0],
    [9.6, 10.7, 10.5],
    [161.2, 164.3, 154.8],
    [156.2, 159.0, 151.4],
    [159.4, 74.6, 87.9],
    [145.9, 53.7, 72.4],
])
mae = np.array([
    [0.0590, 0.0553, 0.0512],
    [0.0344, 0.0360, 0.0311],
    [0.0240, 0.0156, 0.0184],
    [0.0232, 0.0179, 0.0183],
    [0.0574, 0.1136, 0.0918],
    [0.0454, 0.1013, 0.0712],
])
dist = np.array([
    [0.256, 0.277, 0.270],
    [0.230, 0.256, 0.247],
    [0.068, 0.073, 0.086],
    [0.072, 0.079, 0.085],
    [0.147, 0.324, 0.281],
    [0.130, 0.314, 0.257],
])
time_ms = np.array([
    [84.83, 115.17, 94.31],
    [18.71, 21.14, 20.92],
    [163.17, 166.11, 158.84],
    [73.22, 75.45, 73.28],
    [167.98, 77.95, 93.16],
    [71.34, 26.17, 36.06],
])

all_data = [
    (success,  "Success Rate (%)",       [0, 115]),
    (steps,    "Avg. Steps",              None),
    (mae,      "MAE (rad)",              None),
    # (dist,     "Avg. Escape Dist (rad)",  None),
    (time_ms,  "Time (ms)",              None),
]

# ── Style ──
plt.rcParams.update({
    "font.family": "serif",
        "font.size": 36,
        "axes.labelsize": 40,
        "axes.titlesize": 42,
        "xtick.labelsize": 36,
        "ytick.labelsize": 36,
        "legend.fontsize": 36,
})

n_metrics = len(all_data)
fig, axes = plt.subplots(1, n_metrics, figsize=(9 * n_metrics, 10))

colors  = ['#0072B2', '#0072B2', '#E69F00', '#E69F00', '#999999', '#999999']
hatches = ['',        '////',    '',        '////',    '',        '////']

x = np.arange(len(worlds))
n_methods = len(methods)
width = 0.8 / n_methods

for ax_idx, (data, title, ylim) in enumerate(all_data):
    ax = axes[ax_idx]
    for i, method in enumerate(methods):
        offset = (i - (n_methods - 1) / 2) * width
        ax.bar(x + offset, data[i], width,
               label=method if ax_idx == 0 else "",
               color=colors[i],
               hatch=hatches[i],
               edgecolor='black',
               linewidth=1.2)

    ax.set_title(title, pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(worlds)
    if ylim:
        ax.set_ylim(ylim)

    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
    ax.set_axisbelow(True)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper center',
           bbox_to_anchor=(0.5, 1.25), 
           ncol=3,
           fontsize=26, 
           frameon=False,
           handlelength=3.0, 
           handleheight=1.5)

plt.tight_layout()
out_path = "output/escape/escape_comparison.png"
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved to: {out_path}")
