#!/usr/bin/env python3
import argparse
import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import f1_score, confusion_matrix
import random
import robot_tools
import cspace_sdf
from common.models import load_model
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--world', default='block_world1', help="world name")
    parser.add_argument('--n-samples', type=int, default=2000, help="Number of random samples for correlation")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--grid-res', type=int, default=256, help="Resolution for GT SDF calculation")
    args = parser.parse_args()

    seed(args.seed)

    device = get_device()
    print(f"Using {device}")
    
    sense = "config/" + args.world + ".yaml"
    model = "output/" + args.world + "/model_final.pt"
    
    problem = robot_tools.PlanningProblem(sense)
    
    theta, c_space = problem.construct_config_space(nx=args.grid_res)
    theta_periodic, sdf_grid = cspace_sdf.cobs_sdf_on_torus(theta, c_space)
    
    model = load_model(model, device)
    
    print(f"Sampling {args.n_samples} points...")
    qs = np.random.uniform(0, 2*np.pi, (args.n_samples, 2))
    
    qs_wrapped = cspace_sdf.wrap_samples_to_period(qs, theta_periodic)
    d_gt = cspace_sdf.cspace_sdf_at_samples(theta_periodic, sdf_grid, qs_wrapped)
    
    qs_t = torch.from_numpy(qs).float().to(device)
    with torch.no_grad():
        d_pred = model(qs_t).cpu().numpy().flatten()
        
    corr, _ = spearmanr(d_gt, d_pred)
    pearson_corr, _ = pearsonr(d_gt, d_pred)
    
    y_true = d_gt > 0
    y_pred = d_pred > 0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    acc = np.mean(y_true == y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=[False, True])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"CORRELATION ANALYSIS (2D C-Space)")
    print(f"{'='*60}")
    print(f"Points Analyzed:      {len(d_gt)}")
    print(f"Spearman Correlation: {corr:.4f}")
    print(f"Pearson Correlation:  {pearson_corr:.4f}")
    print(f"F1 Score:             {f1:.4f}")
    print(f"Accuracy:             {acc*100:.2f}%")
    print(f"FNR:                  {fnr*100:.2f}%")
    print(f"FPR:                  {fpr*100:.2f}%")
    print(f"{'='*60}\n")
