import torch
import numpy as np
import mujoco
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, confusion_matrix
from common.models import load_model
from geometry_tools import ControlPointManager
import random

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

_GT_MJ_MODEL = None
_GT_MJ_DATA = None
_GT_IS_ROBOT = None
_GT_EXHAUSTIVE = False

def _init_gt_worker(scene_xml, robot_body_indices, exhaustive=False):
    global _GT_MJ_MODEL, _GT_MJ_DATA, _GT_IS_ROBOT, _GT_EXHAUSTIVE
    _GT_MJ_MODEL = mujoco.MjModel.from_xml_path(scene_xml)
    _GT_MJ_DATA = mujoco.MjData(_GT_MJ_MODEL)
    _GT_IS_ROBOT = np.zeros(_GT_MJ_MODEL.nbody, dtype=bool)
    _GT_IS_ROBOT[robot_body_indices] = True
    _GT_EXHAUSTIVE = exhaustive

def _worker_dist_approx(q):
    mj_model, mj_data, is_robot = _GT_MJ_MODEL, _GT_MJ_DATA, _GT_IS_ROBOT
    
    def check_collision(q_test):
        mj_data.qpos[:7] = q_test
        if mj_model.nq > 7:
            mj_data.qpos[7:mj_model.nq] = 0.04
        mujoco.mj_forward(mj_model, mj_data)
        
        # XOR logic: collision only if one geom is robot and the other is environment
        colliding_pairs = []
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
            if is_robot[b1] ^ is_robot[b2]:
                pair = f"{mj_model.body(b1).name}<->{mj_model.body(b2).name}"
                colliding_pairs.append(pair)
        
        if len(colliding_pairs) > 0:
            return True, list(set(colliding_pairs))
        return False, []

    lo, hi = mj_model.jnt_range[:7, 0], mj_model.jnt_range[:7, 1]
    coll_info = check_collision(q)
    is_coll = coll_info[0]
    
    # Initial search schedule
    radii = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    Ns    = [120, 80, 50, 30, 20, 10]
    
    # Anisotropic weights: favor proximal/sensitive joints for higher hit-rate
    weights = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6])
    
    current_min = float('inf')
    found_crossing = False

    for radius, n_samples in zip(radii, Ns):
        for _ in range(n_samples):
            delta = np.random.randn(7) * weights
            delta = (delta / (np.linalg.norm(delta) + 1e-9)) * radius * np.random.rand()
            q_test = np.clip(q + delta, lo, hi)
            
            c_test_info = check_collision(q_test)
            if c_test_info[0] != is_coll:
                found_crossing = True
                q_cross, q_orig = q_test, q
                for _ in range(9):
                    mid = (q_cross + q_orig) / 2
                    if check_collision(mid)[0] != is_coll: q_cross = mid
                    else: q_orig = mid
                d = np.linalg.norm(q_cross - q)
                if d < current_min: current_min = d
        
        if found_crossing and current_min < radius * 0.7:
            break
            
    # Exhaustive mode: if still not hit, keep expanding up to 6.0 rad
    if not found_crossing and _GT_EXHAUSTIVE:
        r_step = 1.6
        while r_step < 6.0 and not found_crossing:
            r_step += 0.8
            for _ in range(20):
                delta = np.random.randn(7) * weights
                delta = (delta / (np.linalg.norm(delta) + 1e-9)) * r_step * np.random.rand()
                q_test = np.clip(q + delta, lo, hi)
                c_test_info = check_collision(q_test)
                if c_test_info[0] != is_coll:
                    found_crossing = True
                    q_cross, q_orig = q_test, q
                    for _ in range(9):
                        mid = (q_cross + q_orig) / 2
                        if check_collision(mid)[0] != is_coll: q_cross = mid
                        else: q_orig = mid
                    current_min = np.linalg.norm(q_cross - q)
                    break

    if not found_crossing:
        return is_coll, np.nan, False, coll_info[1] if is_coll else []

    return is_coll, current_min if not is_coll else -current_min, True, coll_info[1] if is_coll else []

def evaluate_franka_correlation(model_path, scene_xml, n_samples, out_dir, exhaustive=False):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    model.eval()
    
    abs_xml = os.path.abspath(scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
    
    robot_body_indices = []
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    for i in range(mj_model.nbody):
        curr = i
        while curr > 0:
            if curr == base_id:
                robot_body_indices.append(i)
                break
            curr = mj_model.body_parentid[curr]

    print(f"Sampling {n_samples} points in 7D...")
    lo, hi = mj_model.jnt_range[:7, 0], mj_model.jnt_range[:7, 1]
    qs = lo + np.random.rand(n_samples, 7) * (hi - lo)
    
    # 1. Compute NDF values in bulk on GPU
    print("Computing NDF values...")
    
    # Initialize FK preprocessing (matches model setting)
    cp_manager = ControlPointManager(device=device, n_points_per_link=1)
    
    ndf_vals = []
    batch_size = 1000
    for i in range(0, n_samples, batch_size):
        q_batch = torch.from_numpy(qs[i:i+batch_size]).float().to(device)
        
        # Pass joint angles (7D) - model's internal preprocess will handle 21D conversion
        with torch.no_grad():
            ndf_outputs = model(q_batch).squeeze()
            if ndf_outputs.dim() == 0: ndf_outputs = ndf_outputs.unsqueeze(0)
            ndf_vals.extend(ndf_outputs.cpu().numpy().tolist())
    ndf_vals = np.array(ndf_vals)

    # 2. Parallel distance approximation
    print("Computing Ground Truth distances (Optimized Parallel)..." + (" (Exhaustive)" if exhaustive else ""))
    from multiprocessing import Pool
    with Pool(initializer=_init_gt_worker, initargs=(abs_xml, robot_body_indices, exhaustive)) as pool:
        results = list(tqdm(pool.imap(_worker_dist_approx, qs), total=n_samples, desc="GT Analysis"))
    
    gt_labels = np.array([r[0] for r in results])
    approx_dist = np.array([r[1] for r in results])
    hit_flags = np.array([r[2] for r in results])
    coll_pairs_list = [r[3] for r in results]
    
    # Calculate Metrics
    # 1. Spearman Correlation only on HIT samples (Scientific approach)
    hit_mask = hit_flags
    if np.any(hit_mask):
        corr_dist, _ = spearmanr(approx_dist[hit_mask], ndf_vals[hit_mask])
    else:
        corr_dist = 0.0
    
    # 2. Binary correlation (all points - the collision labels are exact)
    corr_binary, _ = spearmanr(ndf_vals, -gt_labels.astype(float))
    
    hit_rate = np.mean(hit_flags) * 100
    
    # Threshold logic
    y_true = gt_labels
    fixed_thresh = getattr(args, "fixed_thresh", None)
    if fixed_thresh is not None:
        best_thresh = fixed_thresh
        y_pred_best = ndf_vals < best_thresh
        best_f1 = f1_score(y_true, y_pred_best, zero_division=0)
        best_acc = np.mean(y_true == y_pred_best)
        print(f"Using fixed threshold: {best_thresh}")
    else:
        # Automatic threshold optimization (maximize F1 score - using all points)
        thresh_range = np.linspace(np.percentile(ndf_vals, 5), np.percentile(ndf_vals, 95), 100)
        best_f1, best_thresh, best_acc = 0, 0, 0
        for t in thresh_range:
            y_pred = ndf_vals < t
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = t
                best_acc = np.mean(y_true == y_pred)
        y_pred_best = ndf_vals < best_thresh
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    # FNR Breakdown
    gripper_keywords = {'hand', 'left_finger', 'right_finger', 'finger'}
    fn_indices = np.where((y_true == True) & (y_pred_best == False))[0]
    fn_gripper = 0
    fn_arm = 0
    arm_link_counts = {}
    
    for idx in fn_indices:
        pairs = coll_pairs_list[idx]
        is_gripper = any(any(kw in p.lower() for kw in gripper_keywords) for p in pairs)
        if is_gripper:
            fn_gripper += 1
        else:
            fn_arm += 1
            # Track which arm links are involved
            for p in pairs:
                # p is "body1<->body2", find the robot one
                parts = p.split("<->")
                for part in parts:
                    if part.startswith("fr3_link") or part == "base":
                        arm_link_counts[part] = arm_link_counts.get(part, 0) + 1
            
    print(f"\n{'='*60}")
    print(f"CORRELATION & CLASSIFICATION ANALYSIS (Franka 7-DOF)")
    print(f"{'='*60}")
    print(f"Points Analyzed:      {n_samples}")
    print(f"GT Hit Rate:          {hit_rate:.2f}%")
    print(f"Distance Correlation: {corr_dist:.4f} (on hit points)")
    print(f"Optimal Threshold:    {best_thresh:.4f}")
    print(f"Best F1 Score:        {best_f1:.4f}")
    print(f"Best Accuracy:        {best_acc*100:.2f}%")
    print(f"FNR (Missed Coll):    {fnr*100:.2f}% ({fn}/{tp+fn})")
    print(f"  - Missed Gripper Coll: {fn_gripper} ({fn_gripper/max(1, fn)*100:.1f}% of FN)")
    print(f"  - Missed Arm Coll:     {fn_arm} ({fn_arm/max(1, fn)*100:.1f}% of FN)")
    if fn_arm > 0:
        sorted_links = sorted(arm_link_counts.items(), key=lambda x: x[1], reverse=True)
        print(f"    [Arm Breakdown]: ", end="")
        for link, count in sorted_links:
            print(f"{link}({count}) ", end="")
        print()
    print(f"FPR (False Alarm):    {fpr*100:.2f}%")
    print(f"{'='*60}\n")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot: Scatter plot with hit/miss distinction
    sns.set_theme()
    plt.figure(figsize=(7, 7))
    
    # 1. Scatter HIT points (True distance known)
    if np.any(hit_mask):
        plt.scatter(approx_dist[hit_mask], ndf_vals[hit_mask], s=12, alpha=0.6, label='Hit (True Distance)')
    
    # 2. Scatter MISS points (Visualized at boundaries)
    miss_mask = ~hit_flags
    if np.any(miss_mask):
        # Place miss points at +/- 1.6 with some jitter
        viz_dist = (1.6 + np.random.rand(np.sum(miss_mask)) * 0.15) * np.where(gt_labels[miss_mask], -1, 1)
        plt.scatter(viz_dist, ndf_vals[miss_mask], s=8, alpha=0.2, color='gray', label='Miss (Out of Range)')
    
    plt.axhline(best_thresh, color='r', linestyle='--', label=f'Optimal Threshold={best_thresh:.3f}')
    plt.axvline(0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel("Approximate GT C-Space Distance (rad)")
    plt.ylabel("NDF Prediction")
    plt.title(f"NDF Evaluation (Franka 7D)\nHit Rate: {hit_rate:.1f}%, Spearman_hit: {corr_dist:.3f}")
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/{args.world}.png", dpi=300)
    print(f"Saved correlation plot to {out_dir}/{args.world}.png")
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output/table_obs10/model_final.pt", help = "model file")
    parser.add_argument("--scene", default="scenes/table_obs10.xml", help = "scene file")
    parser.add_argument("--world", type=str, default="obs10", help = "world name")
    parser.add_argument("--samples", type=int, default=5000, help = "number of samples")
    parser.add_argument("--out", type=str, default="output/correlation", help = "output directory")
    parser.add_argument("--fixed-thresh", type=float, default=None, help="Use a fixed threshold for classification")
    parser.add_argument("--exhaustive", action="store_true", help="Search much further to reach 100% Hit Rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    if args.seed is not None:
        set_seed(args.seed)
    
    evaluate_franka_correlation(args.model, args.scene, args.samples, args.out, exhaustive=args.exhaustive)
