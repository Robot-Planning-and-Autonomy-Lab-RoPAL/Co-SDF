#!/usr/bin/env python3
import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
from glob import glob

def plot_7d_escape_comparison(results_dir, output_path):
    json_files = sorted(glob(os.path.join(results_dir, "results_7d_obs*.json")), 
                        key=lambda x: int(re.search(r'\d+', x).group()))
    
    if not json_files:
        print(f"No results_7d_obs*.json files found in {results_dir}")
        return

    data_map = {}
    all_methods = ["Co-SDF Gradient", "Co-SDF Sampling"]
    
    # Method name mapping from JSON keys to display names
    name_map = {
        "Gradient": "Co-SDF Gradient",
        "Sampling": "Co-SDF Sampling",
    }
    
    for f_path in json_files:
        with open(f_path, 'r') as f:
            data = json.load(f)
        
        world_name = data.get("world", os.path.basename(f_path)).replace(".json", "")
        summary = data.get("summary", {})
        details = data.get("details", {})
        
        world_data = {}
        for key, metrics in summary.items():
            # Try new format first (plain name like "Gradient")
            display_name = name_map.get(key, None)
            
            # Fallback: old format "Gradient (M=0.02)"
            if display_name is None:
                match = re.search(r"([\w\s]+) \(M=([\d\.]+)\)", key)
                if match:
                    method_base = match.group(1).strip()
                    display_name = name_map.get(method_base, method_base)
            
            if display_name is None:
                continue

            # Re-calculate avg_steps and avg_dist for success only
            method_details = details.get(key, [])
            success_trials = [d for d in method_details if d.get("success", False)]
            if success_trials:
                metrics["avg_steps"] = np.mean([d["steps"] for d in success_trials])
                metrics["avg_escape_dist"] = np.mean([d["dist"] for d in success_trials])
            
            world_data[display_name] = metrics
        
        data_map[world_name] = world_data

    worlds = sorted(data_map.keys(), key=lambda x: int(re.search(r'\d+', x).group()))

    metrics_info = [
        ("success_rate", "Success Rate (%)", [0, 110]),
        ("avg_steps", "Avg. Escape Steps", None),
        ("mae_vs_gt", "MAE (rad)", None),
        ("avg_gt_dist", "Avg. C-space Disp (rad)", None),
        ("avg_time_ms", "Time (ms)", None)
    ]

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 36,
        "axes.labelsize": 40,
        "axes.titlesize": 42,
        "xtick.labelsize": 36,
        "ytick.labelsize": 36,
        "legend.fontsize": 36,
    })

    n_metrics = len(metrics_info)
    fig, axes = plt.subplots(1, n_metrics, figsize=(9 * n_metrics, 10))
    if n_metrics == 1:
        axes = [axes]
    
    colors = ['#0072B2', '#0072B2'] 
    hatches = ['', '////']

    x = np.arange(len(worlds))
    width = 0.35

    for idx, (m_key, title, ylim) in enumerate(metrics_info):
        ax = axes[idx]
        for i, method in enumerate(all_methods):
            vals = [data_map[w].get(method, {}).get(m_key, 0.0) for w in worlds]
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
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.15), 
               ncol=2, fontsize=32, frameon=False,
               handlelength=3.0, handleheight=1.5)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 7D ESCAPE comparison figure to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='output/escape')
    parser.add_argument('--output', default='output/escape/escape.png')
    args = parser.parse_args()
    plot_7d_escape_comparison(args.results_dir, args.output)
