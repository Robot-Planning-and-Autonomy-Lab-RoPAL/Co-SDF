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

_GT_MJ_MODEL = None
_GT_MJ_DATA = None
_GT_IS_ROBOT = None

def _init_gt_worker(scene_xml, robot_body_indices):
    global _GT_MJ_MODEL, _GT_MJ_DATA, _GT_IS_ROBOT
    _GT_MJ_MODEL = mujoco.MjModel.from_xml_path(scene_xml)
    _GT_MJ_DATA = mujoco.MjData(_GT_MJ_MODEL)
    _GT_IS_ROBOT = np.zeros(_GT_MJ_MODEL.nbody, dtype=bool)
    _GT_IS_ROBOT[robot_body_indices] = True

def _worker_dist_approx(q):
    mj_model, mj_data, is_robot = _GT_MJ_MODEL, _GT_MJ_DATA, _GT_IS_ROBOT
    
    def check_collision(q_test):
        mj_data.qpos[:7] = q_test
        if mj_model.nq > 7:
            mj_data.qpos[7:mj_model.nq] = 0.04
        mujoco.mj_forward(mj_model, mj_data)
        return mj_data.ncon > 0

    lo, hi = mj_model.jnt_range[:7, 0], mj_model.jnt_range[:7, 1]
    is_coll = check_collision(q)
    
    # User-recommended exponential schedule and decreasing budget
    radii = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    Ns    = [120, 80, 50, 30, 20, 10]
    
    # Anisotropic weights: favor proximal/sensitive joints for higher hit-rate
    weights = np.array([1.0, 1.0, 1.0, 0.8, 0.8, 0.6, 0.6])
    
    current_min = float('inf')
    found_crossing = False

    for radius, n_samples in zip(radii, Ns):
        for _ in range(n_samples):
            # Anisotropic direction sampling
            delta = np.random.randn(7) * weights
            # Scale to radius * u for volume sampling
            delta = (delta / (np.linalg.norm(delta) + 1e-9)) * radius * np.random.rand()
            q_test = np.clip(q + delta, lo, hi)
            
            if check_collision(q_test) != is_coll:
                found_crossing = True
                q_cross, q_orig = q_test, q
                # Binary search refinement (9 steps for high precision)
                for _ in range(9):
                    mid = (q_cross + q_orig) / 2
                    if check_collision(mid) != is_coll: q_cross = mid
                    else: q_orig = mid
                
                d = np.linalg.norm(q_cross - q)
                if d < current_min:
                    current_min = d
        
        # Aggressive early exit: if we found a point much closer than the current radius,
        # we've likely captured the local boundary.
        if found_crossing and current_min < radius * 0.7:
            break
            
    if not found_crossing:
        # Miss: return NaN for distance to avoid corrupting Spearman rank
        return is_coll, np.nan, False

    return is_coll, current_min if not is_coll else -current_min, True

def evaluate_franka_correlation(model_path, scene_xml, n_samples, out_dir):
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
    print("Computing Ground Truth distances (Optimized Parallel)...")
    from multiprocessing import Pool
    with Pool(initializer=_init_gt_worker, initargs=(abs_xml, robot_body_indices)) as pool:
        results = list(tqdm(pool.imap(_worker_dist_approx, qs), total=n_samples, desc="GT Analysis"))
    
    gt_labels = np.array([r[0] for r in results])
    approx_dist = np.array([r[1] for r in results])
    hit_flags = np.array([r[2] for r in results])
    
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
    
    # Automatic threshold optimization (maximize F1 score - using all points)
    y_true = gt_labels
    thresh_range = np.linspace(np.percentile(ndf_vals, 5), np.percentile(ndf_vals, 95), 100)
    
    best_f1, best_thresh, best_acc = 0, 0, 0
    for t in thresh_range:
        y_pred = ndf_vals < t  # collision if NDF < threshold
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t
            best_acc = np.mean(y_true == y_pred)
    
    # Evaluate with optimal threshold
    y_pred_best = ndf_vals < best_thresh
    cm = confusion_matrix(y_true, y_pred_best)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    print(f"\n{'='*60}")
    print(f"CORRELATION & CLASSIFICATION ANALYSIS (Franka 7-DOF)")
    print(f"{'='*60}")
    print(f"Points Analyzed:      {n_samples}")
    print(f"GT Hit Rate:          {hit_rate:.2f}%")
    print(f"Distance Correlation: {corr_dist:.4f} (on hit points)")
    print(f"Binary Correlation:   {corr_binary:.4f}")
    print(f"Optimal Threshold:    {best_thresh:.4f}")
    print(f"Best F1 Score:        {best_f1:.4f}")
    print(f"Best Accuracy:        {best_acc*100:.2f}%")
    print(f"FNR (Missed Coll):    {fnr*100:.2f}%")
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
    
    plt.savefig(f"{out_dir}/correlation.png", dpi=300)
    print(f"Saved correlation plot to {out_dir}/correlation.png")
    plt.close()


def verify_gradients(model_path, scene_xml, n_samples=100):
    """Verify that NDF gradients point away from obstacles"""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, device)
    model.eval()
    
    # Initialize FK preprocessing
    cp_manager = ControlPointManager(device=device, n_points_per_link=2)
    
    abs_xml = os.path.abspath(scene_xml)
    mj_model = mujoco.MjModel.from_xml_path(abs_xml)
    mj_data = mujoco.MjData(mj_model)
    
    robot_body_ids = set()
    base_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "base")
    for i in range(mj_model.nbody):
        curr = i
        while curr > 0:
            if curr == base_id:
                robot_body_ids.add(i)
                break
            curr = mj_model.body_parentid[curr]
    is_robot_body = np.zeros(mj_model.nbody, dtype=bool)
    if robot_body_ids:
        is_robot_body[list(robot_body_ids)] = True

    def check_collision(q):
        mj_data.qpos[:7] = q
        mujoco.mj_forward(mj_model, mj_data)
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
            if is_robot_body[b1] ^ is_robot_body[b2]:
                return True
        return False
    
    lo, hi = mj_model.jnt_range[:7, 0], mj_model.jnt_range[:7, 1]
    
    print(f"\nVerifying gradients on {n_samples} boundary points...")
    correct_grads = 0
    total_boundary = 0
    grad_angles = []
    
    for _ in tqdm(range(n_samples)):
        # Sample a free point
        q = lo + np.random.rand(7) * (hi - lo)
        while check_collision(q):
            q = lo + np.random.rand(7) * (hi - lo)
        
        # Find nearby collision point (boundary)
        for trial in range(50):
            delta = np.random.randn(7) * 0.05
            q_coll = np.clip(q + delta, lo, hi)
            if check_collision(q_coll):
                # Found boundary! Check gradient
                q_t = torch.from_numpy(q_coll).float().unsqueeze(0).to(device)
                q_t.requires_grad = True
                
                # Pass joint angles (7D) - model's internal preprocess will handle 21D conversion
                ndf_val = model(q_t)
                ndf_val.backward()
                grad = q_t.grad.squeeze().cpu().numpy()
                
                # Gradient should point from collision (q_coll) toward free (q)
                toward_free = q - q_coll
                toward_free = toward_free / (np.linalg.norm(toward_free) + 1e-8)
                grad_norm = grad / (np.linalg.norm(grad) + 1e-8)
                
                # Cosine similarity: should be positive
                cos_sim = np.dot(grad_norm, toward_free)
                grad_angles.append(cos_sim)
                
                if cos_sim > 0:
                    correct_grads += 1
                total_boundary += 1
                break
    
    grad_angles = np.array(grad_angles)
    
    print(f"\n{'='*60}")
    print(f"GRADIENT VERIFICATION (Franka 7-DOF)")
    print(f"{'='*60}")
    print(f"Boundary Points:      {total_boundary}")
    print(f"Correct Direction:    {correct_grads} ({100*correct_grads/max(total_boundary,1):.1f}%)")
    print(f"Mean Cosine Sim:      {grad_angles.mean():.4f}")
    print(f"Median Cosine Sim:    {np.median(grad_angles):.4f}")
    print(f"{'='*60}\n")
    
    # Plot gradient alignment
    out_dir = "output/correlation_franka"
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.hist(grad_angles, bins=50, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Alignment')
    plt.axvline(grad_angles.mean(), color='green', linestyle='-', linewidth=2, label=f'Mean={grad_angles.mean():.3f}')
    plt.xlabel('Cosine Similarity (Gradient · Toward-Free-Direction)')
    plt.ylabel('Count')
    plt.title('NDF Gradient Alignment with Free Space Direction\n(>0 is correct)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/gradient_verification.png", dpi=300)
    print(f"Gradient verification plot saved to {out_dir}/gradient_verification.png")
    plt.close()
    
    return grad_angles


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="output/bookshelf/model_final.pt")
    parser.add_argument("--scene", default="scenes/bookshelf.xml")
    parser.add_argument("--samples", type=int, default=2000)
    parser.add_argument("--verify-grad", action="store_true", help="Also verify gradients")
    parser.add_argument("--grad-samples", type=int, default=100)
    parser.add_argument("--out", type=str, default="output/correlation")
    args = parser.parse_args()
    
    evaluate_franka_correlation(args.model, args.scene, args.samples, args.out)
    
    if args.verify_grad:
        verify_gradients(args.model, args.scene, args.grad_samples)
