#!/usr/bin/env python3
import os
import json
import glob
import numpy as np
import matplotlib.pyplot as plt
import argparse
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Plot Gradient Escape Results Histogram")
    parser.add_argument('--results-dir', default='output/escape_sampling', help="Directory containing result JSON files")
    parser.add_argument('--output-dir', default='output/escape_sampling', help="Directory to save output plots")
    return parser.parse_args()

def extract_obstacle_count(filename):
    # Try to find a number in the filename, assuming format like 'block_world5'
    match = re.search(r'block_world(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def main():
    args = parse_args()
    
    json_files = glob.glob(os.path.join(args.results_dir, "*.json"))
    if not json_files:
        print(f"No JSON files found in {args.results_dir}")
        return

    data = []
    for fpath in json_files:
        try:
            with open(fpath, 'r') as f:
                res = json.load(f)
                
            # Extract obstacle count from world name or filename
            world_name = res.get('world', '')
            obs_count = extract_obstacle_count(world_name)
            if obs_count is None:
                obs_count = extract_obstacle_count(os.path.basename(fpath))
                
            if obs_count is not None:
                total_time = res.get('total_time', 0)
                n_samples = res.get('n_samples', 1)
                data.append({
                    'obstacles': obs_count,
                    'gt_success': res['gt_success_rate'],
                    'ndf_success': res['ndf_success_rate'],
                    'avg_steps': res.get('avg_gt_steps', 0),
                    'avg_time_ms': (total_time / n_samples) * 1000 if n_samples > 0 else 0
                })
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    if not data:
        print("No valid data extracted.")
        return

    # Sort by obstacle count
    data.sort(key=lambda x: x['obstacles'])
    
    obstacles = [str(d['obstacles']) for d in data]
    x = np.arange(len(obstacles))
    width = 0.5

    # 1. GT Success Rate
    gt_success = [d['gt_success'] * 100 for d in data]
    fig, ax = plt.subplots(figsize=(8, 6))
    rects = ax.bar(x, gt_success, width, color='skyblue', label='GT Success')
    ax.set_xlabel('Number of Obstacles')
    ax.set_ylabel('Success Rate (%)')
    ax.set_title('Gradient Escape Success Rate (Collision Free)')
    ax.set_xticks(x)
    ax.set_xticklabels(obstacles)
    ax.bar_label(rects, padding=3, fmt='%.1f')
    ax.set_ylim(0, 110)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "success_rate.png"))
    plt.close()
    print(f"Saved success_rate.png")

    # 2. Avg Steps to GT
    avg_steps = [d['avg_steps'] for d in data]
    fig, ax = plt.subplots(figsize=(8, 6))
    rects = ax.bar(x, avg_steps, width, color='lightgreen', label='Avg Steps')
    ax.set_xlabel('Number of Obstacles')
    ax.set_ylabel('Average Steps')
    ax.set_title('Average Steps to Reach Free Space')
    ax.set_xticks(x)
    ax.set_xticklabels(obstacles)
    ax.bar_label(rects, padding=3, fmt='%.1f')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "avg_steps_gt.png"))
    plt.close()
    print(f"Saved avg_steps_gt.png")

    # 3. Avg Time / Sample
    avg_times = [d['avg_time_ms'] for d in data]
    fig, ax = plt.subplots(figsize=(8, 6))
    rects = ax.bar(x, avg_times, width, color='salmon', label='Avg Time')
    ax.set_xlabel('Number of Obstacles')
    ax.set_ylabel('Time per Sample (ms)')
    ax.set_title('Average Computation Time per Sample')
    ax.set_xticks(x)
    ax.set_xticklabels(obstacles)
    ax.bar_label(rects, padding=3, fmt='%.1f')
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "avg_time_sample.png"))
    plt.close()
    print(f"Saved avg_time_sample.png")

if __name__ == "__main__":
    main()
