#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
from glob import glob

def plot_7d_escape_comparison(results_dir, output_path, target_margin=0.02):
    json_files = sorted(glob(os.path.join(results_dir, "results_7d_obs*.json")), 
                        key=lambda x: int(re.search(r'\d+', x).group()))
    
    if not json_files:
        print(f"No results_7d_obs*.json files found in {results_dir}")
        return

    data_map = {}
    all_margins = set()
    all_methods = ["Gradient", "Sampling"]
    
    for f_path in json_files:
        with open(f_path, 'r') as f:
            data = json.load(f)
        
        world_name = data.get("world", os.path.basename(f_path)).replace(".json", "")
        summary = data.get("summary", {})
        
        world_data = {}
        for key, metrics in summary.items():
            match = re.search(r"([\w\s]+) \(M=([\d\.]+)\)", key)
            if match:
                method_base = match.group(1).strip()
                margin = float(match.group(2))
                all_margins.add(margin)
                if margin not in world_data: world_data[margin] = {}
                world_data[margin][method_base] = metrics
        
        data_map[world_name] = world_data

    worlds = sorted(data_map.keys(), key=lambda x: int(re.search(r'\d+', x).group()))
    
    margins = sorted(list(all_margins))
    if not any(np.isclose(target_margin, m) for m in margins):
        target_margin = margins[0] if margins else 0.02

    metrics_info = [
        ("gt_rate", "Success Rate (%)", [0, 110]),
        ("avg_ndf_steps", "Avg. Escape Steps", None),
        ("avg_dist", "Avg. Joint Displacement (rad)", None),
        ("avg_time_ms", "Escape Time (ms)", None)
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

    fig, axes = plt.subplots(1, 4, figsize=(28, 7))
    
    colors = ['#0072B2', '#E69F00'] 
    hatches = ['', '']

    x = np.arange(len(worlds))
    width = 0.35

    for idx, (m_key, title, ylim) in enumerate(metrics_info):
        ax = axes[idx]
        for i, method in enumerate(all_methods):
            vals = [data_map[w].get(target_margin, {}).get(method, {}).get(m_key, 0.0) for w in worlds]
            offset = (i - 0.5) * width
            
            ax.bar(x + offset, vals, width, 
                   label=method if idx == 0 else "", 
                   color=colors[i], 
                   hatch=hatches[i],
                   edgecolor='black', 
                   linewidth=1.2)

        ax.set_title(title, pad=20, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{w.replace('obs', '')} Obs" for w in worlds])
        if ylim: ax.set_ylim(ylim)
        
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=0.3)
        ax.set_axisbelow(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), 
               ncol=2, fontsize=18, frameon=False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 7D ESCAPE comparison figure to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='output/escape')
    parser.add_argument('--target-margin', type=float, default=0.02)
    parser.add_argument('--output', default='output/escape/escape.png')
    args = parser.parse_args()
    plot_7d_escape_comparison(args.results_dir, args.output, args.target_margin)
