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
        mujoco.mj_forward(mj_model, mj_data)
        for i in range(mj_data.ncon):
            c = mj_data.contact[i]
            b1, b2 = mj_model.geom_bodyid[c.geom1], mj_model.geom_bodyid[c.geom2]
            if is_robot[b1] or is_robot[b2]: return True
        return False

    lo, hi = mj_model.jnt_range[:7, 0], mj_model.jnt_range[:7, 1]
    is_coll = check_collision(q)
    
    refine_samples = 50
    max_dist = 0.5
    current_min = max_dist
    
    if is_coll:
        for _ in range(refine_samples):
            delta = np.random.randn(7) * 0.1
            q_test = np.clip(q + delta, lo, hi)
            if not check_collision(q_test):
                d = np.linalg.norm(delta)
                if d < current_min:
                    q_free, q_orig = q_test, q
                    for _ in range(5):
                        mid = (q_free + q_orig) / 2
                        if not check_collision(mid): q_free = mid
                        else: q_orig = mid
                    current_min = np.linalg.norm(q_free - q)
        return is_coll, -current_min
    else:
        for _ in range(refine_samples):
            delta = np.random.randn(7) * 0.1
            q_test = np.clip(q + delta, lo, hi)
            if check_collision(q_test):
                d = np.linalg.norm(delta)
                if d < current_min:
                    q_coll, q_orig = q_test, q
                    for _ in range(5):
                        mid = (q_coll + q_orig) / 2
                        if check_collision(mid): q_coll = mid
                        else: q_orig = mid
                    current_min = np.linalg.norm(q_coll - q)
        return is_coll, current_min

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
    
    # Initialize FK preprocessing (for models that need it)
    cp_manager = ControlPointManager(device=device, n_points_per_link=2)
    
    ndf_vals = []
    batch_size = 1000
    for i in range(0, n_samples, batch_size):
        q_batch = torch.from_numpy(qs[i:i+batch_size]).float().to(device)
        
        # Compute FK features: control points from joint angles
        with torch.no_grad():
            points = cp_manager.get_control_points(q_batch)  # (B, 14, 3)
            points_flat = points.reshape(q_batch.shape[0], -1)  # (B, 42)
            x_features = torch.cat([q_batch, points_flat], dim=-1)  # (B, 49)
            
            ndf_outputs = model(x_features).squeeze()
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
    
    corr_dist, _ = spearmanr(approx_dist, ndf_vals)
    corr_binary, _ = spearmanr(ndf_vals, -gt_labels.astype(float))
    
    # Automatic threshold optimization (maximize F1 score)
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
    print(f"Distance Correlation: {corr_dist:.4f}")
    print(f"Binary Correlation:   {corr_binary:.4f}")
    print(f"Optimal Threshold:    {best_thresh:.4f}")
    print(f"Best F1 Score:        {best_f1:.4f}")
    print(f"Best Accuracy:        {best_acc*100:.2f}%")
    print(f"FNR (Missed Coll):    {fnr*100:.2f}%")
    print(f"FPR (False Alarm):    {fpr*100:.2f}%")
    print(f"{'='*60}\n")
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Plot: Scatter plot (same style as 2D)
    sns.set_theme()
    plt.figure(figsize=(6, 6))
    plt.scatter(approx_dist, ndf_vals, s=10, alpha=0.5)
    plt.axhline(best_thresh, color='r', linestyle='--', label=f'Optimal Threshold={best_thresh:.3f}')
    plt.axvline(0, color='k', linestyle='-', alpha=0.3)
    plt.xlabel("Approximate GT C-Space Distance (rad)")
    plt.ylabel("NDF Prediction")
    plt.title(f"NDF vs. Approx C-space Distance (7D Franka)")
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
            if is_robot_body[b1] or is_robot_body[b2]:
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
                
                # Compute FK features
                points = cp_manager.get_control_points(q_t)
                points_flat = points.reshape(q_t.shape[0], -1)
                x_features = torch.cat([q_t, points_flat], dim=-1)
                
                ndf_val = model(x_features)
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
