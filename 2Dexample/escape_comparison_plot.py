import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

def plot_comparison(results_path, output_dir):
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found.")
        return

    with open(results_path, 'r') as f:
        data = json.load(f)

    summary = data['summary']
    all_keys = list(summary.keys())
    
    configs = []
    for k in all_keys:
        match = re.search(r"(\w+) (\w+) \(M=([\d\.]+)\)", k)
        if match:
            configs.append({
                "key": k,
                "model": match.group(1),
                "method": match.group(2),
                "margin": float(match.group(3))
            })
    
    if not configs:
        print("Error: Could not parse results keys for grouped plotting.")
        return

    margins = sorted(list(set(c["margin"] for c in configs)))
    groups = [
        ("LIP", "Grad"), ("LIP", "Samp"),
        ("SIREN", "Grad"), ("SIREN", "Samp")
    ]
    
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))
    axes = axes.flatten()
    
    metrics = [
        ("gt_rate", "Success Rate", "% Success", [0, 110]),
        ("avg_gt_steps", "Avg. Steps", "Steps", None),
        ("avg_dist", "Avg. Escape Displacement", "Dist (rad)", None),
        ("avg_time_ms", "Computation Time", "Time (ms)", None)
    ]
    
    colors = ['#2980b9', '#3498db', '#c0392b', '#e74c3c']
    labels = ["LIP Gradient", "LIP Sampling", "SIREN Gradient", "SIREN Sampling"]

    x = np.arange(len(margins))
    width = 0.18

    for idx, (m_key, title, ylabel, ylim) in enumerate(metrics):
        ax = axes[idx]
        
        bars = []
        for i, (model, method) in enumerate(groups):
            vals = []
            for m in margins:
                key = next((c["key"] for c in configs if c["model"] == model and c["method"] == method and c["margin"] == m), None)
                vals.append(summary[key][m_key] if key else 0)
            
            offset = (i - 1.5) * width
            b = ax.bar(x + offset, vals, width, color=colors[i], edgecolor='white', linewidth=0.5, label=labels[i])
            bars.append(b)

        ax.set_title(title, fontsize=15, fontweight='bold', pad=15)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels([f"m={m}" for m in margins], fontsize=10)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        
        if m_key == "avg_dist":
            for i, rect_group in enumerate(bars):
                for rect in rect_group:
                    height = rect.get_height()
                    ax.annotate(f'{height:.2f}',
                                xy=(rect.get_x() + rect.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', 
                                va='bottom', 
                                fontsize=9)

    handles, plot_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, plot_labels, loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=4, fontsize=14, frameon=False)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"{args.world}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved consolidated horizontal figure to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='block_world20', help="World name")
    parser.add_argument('--results', default='output/escape/results_block_world20.json', help="Path to results.json")
    parser.add_argument('--output-dir', default='output/escape', help="Directory to save the plots")
    args = parser.parse_args()
    plot_comparison(args.results, args.output_dir)
