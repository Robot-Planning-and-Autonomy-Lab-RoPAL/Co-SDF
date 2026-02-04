import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
from glob import glob

def plot_obs_comparison(results_dir, output_path, target_margin=0.01):
    # Find all results files
    json_files = sorted(glob(os.path.join(results_dir, "results_block_world*.json")), 
                        key=lambda x: int(re.search(r'\d+', x).group()))
    
    if not json_files:
        print(f"No results_block_world*.json files found in {results_dir}")
        return

    data_map = {}
    all_margins = set()
    all_methods = ["LIP Grad", "LIP Samp", "SIREN Grad", "SIREN Samp"]
    
    for f_path in json_files:
        with open(f_path, 'r') as f:
            data = json.load(f)
        
        world_name = data.get("world", os.path.basename(f_path)).replace(".yaml", "")
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
    
    # Check if target_margin exists, else pick closest or first
    margins = sorted(list(all_margins))
    if target_margin not in margins:
        print(f"Warning: Margin {target_margin} not found. Available: {margins}. Using {margins[0]}")
        target_margin = margins[0]

    metrics_info = [
        ("gt_rate", "Success Rate - 2DOF", [0, 110]),
        ("avg_gt_steps", "Avg. Steps - 2DOF", None),
        ("avg_dist", "Avg. Displacement - 2DOF", None),
        ("avg_time_ms", "Computation Time - 2DOF", None)
    ]

    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    colors = ['#1f77b4', '#ff7f0e', '#9467bd', '#4d4d4d'] 
    x = np.arange(len(worlds))
    width = 0.18

    for idx, (m_key, title, ylim) in enumerate(metrics_info):
        ax = axes[idx]
        for i, method in enumerate(all_methods):
            vals = [data_map[w].get(target_margin, {}).get(method, {}).get(m_key, 0) for w in worlds]
            offset = (i - 1.5) * width
            ax.bar(x + offset, vals, width, label=method, color=colors[i], edgecolor='white', linewidth=0.5)

        ax.set_title(f"{title}", fontsize=16, fontweight='normal', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(["5obs", "10obs", "20obs"], fontsize=14)
        if ylim: ax.set_ylim(ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=14, frameon=False)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved consolidated Obs comparison figure to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='output/escape')
    parser.add_argument('--target-margin', type=float, default=0.005)
    parser.add_argument('--output', default='output/escape/obs_comparison_final.png')
    args = parser.parse_args()
    plot_obs_comparison(args.results_dir, args.output, args.target_margin)
