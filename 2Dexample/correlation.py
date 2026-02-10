#!/usr/bin/env python3
import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, confusion_matrix
import random
import robot_tools
import cspace_sdf
from common.models import load_model as global_load_model
from common.utils import get_device

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def load_model(path, device, n_layers, n_hidden):
    return global_load_model(path, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='obs10', help="world name")
    parser.add_argument('--n-samples', type=int, default=3000, help="Number of random samples for correlation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--grid-res', type=int, default=256, help="Resolution for GT SDF calculation")
    parser.add_argument('--bound-limit', type=float, default=0.5, help="Distance range (rad) for correlation analysis")
    parser.add_argument('--output-dir', default='output/correlation', help="Output directory for results")
    parser.add_argument('--n-layers', type=int, default=10, help="NDF layers")
    parser.add_argument('--n-hidden', type=int, default=256, help="NDF hidden units")
    args = parser.parse_args()

    seed(args.seed)

    device = get_device()
    print(f"Using {device}")
    
    sense = "config/" + args.world + ".yaml"
    model = "output/" + args.world + "_siren/model_final.pt"
    
    problem = robot_tools.PlanningProblem(sense)
    
    theta, c_space = problem.construct_config_space(nx=args.grid_res)
    theta_periodic, sdf_grid = cspace_sdf.cobs_sdf_on_torus(theta, c_space)
    
    model = load_model(model, device, n_layers=args.n_layers, n_hidden=args.n_hidden)
    
    print(f"Sampling {args.n_samples} points...")
    qs = np.random.uniform(0, 2*np.pi, (args.n_samples, 2))
    
    qs_wrapped = cspace_sdf.wrap_samples_to_period(qs, theta_periodic)
    d_gt = cspace_sdf.cspace_sdf_at_samples(theta_periodic, sdf_grid, qs_wrapped)
    
    qs_t = torch.from_numpy(qs).float().to(device)
    with torch.no_grad():
        d_pred = model(qs_t).cpu().numpy().flatten()
        
    mask = (d_gt > -args.bound_limit) & (d_gt < args.bound_limit)
    d_gt_filt = d_gt[mask]
    d_pred_filt = d_pred[mask]
    
    if len(d_gt_filt) < 10:
        print("Not enough points in filter range for analysis. Using all points.")
        d_gt_filt, d_pred_filt = d_gt, d_pred
        mask = np.ones_like(d_gt, dtype=bool)

    corr, p = spearmanr(d_gt_filt, d_pred_filt)
    
    y_true = d_gt_filt > 0
    
    best_f1, best_thresh, best_acc = 0, 0, 0
    thresh_range = np.linspace(np.percentile(d_pred_filt, 5), np.percentile(d_pred_filt, 95), 100)
    for t in thresh_range:
        y_pred = d_pred_filt > t
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_acc = np.mean(y_true == y_pred)
            
    y_pred_best = d_pred_filt > best_thresh
    cm = confusion_matrix(y_true, y_pred_best, labels=[False, True])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0,0,0,0)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS (2D C-Space)")
    print(f"{'='*60}")
    print(f"Points Analyzed:      {len(d_gt_filt)}")
    print(f"Spearman Correlation: {corr:.4f}")
    print(f"Optimal Threshold:    {best_thresh:.4f}")
    print(f"Best F1 Score:        {best_f1:.4f}")
    print(f"Best Accuracy:        {best_acc*100:.2f}%")
    print(f"FNR (Missed Coll):    {fnr*100:.2f}%")
    print(f"FPR (False Alarm):    {fpr*100:.2f}%")
    print(f"{'='*60}\n")
    
    os.makedirs(args.output_dir, exist_ok=True)
    sns.set_theme()
    
    plt.figure(figsize=(6, 6))
    plt.scatter(d_gt_filt, d_pred_filt, s=10, alpha=0.5)
    plt.axhline(best_thresh, color='r', linestyle='--', label=f'Feasibility Threshold={best_thresh:.3f}')
    plt.axvline(0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel("Ground Truth C-Space SDF (rad)")
    plt.ylabel("NDF Prediction")
    plt.title(f"NDF vs. Ground-Truth C-space SDF (2D Block World) \nSpearman Correlation: {corr:.4f}")
    plt.legend()
    plt.tight_layout()
    
    plot_path = os.path.join(args.output_dir, f"correlation_{args.world}.png")
    plt.savefig(plot_path, dpi=300)
    print(f"Saved correlation plot to {plot_path}")
    
    plt.close('all')
