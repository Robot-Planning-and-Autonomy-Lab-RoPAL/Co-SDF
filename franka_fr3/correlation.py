import torch
import argparse
import mujoco
import os
import random
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score, confusion_matrix
from common.models import load_model
from common.utils import get_device
from utils import ControlPoint

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
    # Use a fixed truncation radius of 1.6 rad
    R_max = 1.6
    coll_info = check_collision(q)
    is_coll = coll_info[0]
    
    radii = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    Ns    = [120, 80, 50, 30, 20, 10]
    
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
            
    if not found_crossing:
        return is_coll, np.nan, False, coll_info[1] if is_coll else []

    return is_coll, current_min if not is_coll else -current_min, True, coll_info[1] if is_coll else []

def evaluate_franka_correlation(model_path, scene_xml, n_samples, out_dir, device):

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

    print(f"Mining {n_samples} valid points in 7D...")
    lo, hi = mj_model.jnt_range[:7, 0], mj_model.jnt_range[:7, 1]
    
    valid_qs = []
    valid_results = []
    
    from multiprocessing import Pool
    with Pool(initializer=_init_gt_worker, initargs=(abs_xml, robot_body_indices)) as pool:
        pbar = tqdm(total=n_samples, desc="GT Analysis (Mining valid points)")
        while len(valid_results) < n_samples:
            needed = n_samples - len(valid_results)
            batch_size = max(100, int(needed * 1.5))
            qs_batch = lo + np.random.rand(batch_size, 7) * (hi - lo)
            
            for q, res in zip(qs_batch, pool.imap(_worker_dist_approx, qs_batch)):
                if res[2]: # hit_flag is True
                    valid_qs.append(q)
                    valid_results.append(res)
                    pbar.update(1)
                    if len(valid_results) >= n_samples:
                        break
        pbar.close()
        
    qs = np.array(valid_qs)
    
    ndf_vals = []
    batch_size_eval = 1000
    for i in range(0, n_samples, batch_size_eval):
        q_batch = torch.from_numpy(qs[i:i+batch_size_eval]).float().to(device)
        
        with torch.no_grad():
            ndf_outputs = model(q_batch).squeeze()
            if ndf_outputs.dim() == 0: ndf_outputs = ndf_outputs.unsqueeze(0)
            ndf_vals.extend(ndf_outputs.cpu().numpy().tolist())
    ndf_vals = np.array(ndf_vals)
    
    gt_labels = np.array([r[0] for r in valid_results])
    approx_dist = np.array([r[1] for r in valid_results])
    coll_pairs_list = [r[3] for r in valid_results]
    
    corr_dist_spearman, _ = spearmanr(approx_dist, ndf_vals)
    corr_dist_pearson, _ = pearsonr(approx_dist, ndf_vals)
    
    y_true = gt_labels
    fixed_thresh = getattr(args, "fixed_thresh", None)
    if fixed_thresh is not None:
        y_pred_best = ndf_vals < fixed_thresh
        best_f1 = f1_score(y_true, y_pred_best, zero_division=0)
        best_acc = np.mean(y_true == y_pred_best)
        print(f"Using fixed threshold: {fixed_thresh}")
    else:
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
            for p in pairs:
                parts = p.split("<->")
                for part in parts:
                    if part.startswith("fr3_link") or part == "base":
                        arm_link_counts[part] = arm_link_counts.get(part, 0) + 1
            
    print(f"\n{'='*60}")
    print(f"CORRELATION & CLASSIFICATION ANALYSIS (Franka 7-DOF)")
    print(f"{'='*60}")
    print(f"Spearman Correlation:  {corr_dist_spearman:.4f}")
    print(f"Pearson Correlation:   {corr_dist_pearson:.4f}")
    print(f"F1 Score:        {best_f1:.4f}")
    print(f"Accuracy:        {best_acc*100:.2f}%")
    print(f"FNR:    {fnr*100:.2f}%")
    # print(f"  - Missed Gripper Coll: {fn_gripper} ({fn_gripper/max(1, fn)*100:.1f}% of FN)")
    # print(f"  - Missed Arm Coll:     {fn_arm} ({fn_arm/max(1, fn)*100:.1f}% of FN)")
    # if fn_arm > 0:
    #     sorted_links = sorted(arm_link_counts.items(), key=lambda x: x[1], reverse=True)
    #     print(f"    [Arm Breakdown]: ", end="")
    #     for link, count in sorted_links:
    #         print(f"{link}({count}) ", end="")
    #     print()
    print(f"FPR:    {fpr*100:.2f}%")
    print(f"{'='*60}\n")
    
    os.makedirs(out_dir, exist_ok=True)
    
    sns.set_theme()
    plt.figure(figsize=(6, 6))
    
    plt.scatter(approx_dist, ndf_vals, s=10, alpha=0.5)
    
    plt.xlabel("Boundary-Distance Proxy", fontsize=18)
    plt.ylabel("NDF Prediction", fontsize=18)
    plt.title("NDF vs. Boundary-Distance Proxy", fontsize=20)
    plt.tight_layout()
    
    plt.savefig(f"{out_dir}/{args.world}.png", dpi=300)
    print(f"Saved correlation plot to {out_dir}/{args.world}.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", type=str, default="obs10", help = "world name")
    parser.add_argument("--samples", type=int, default=2000, help = "number of samples")
    parser.add_argument("--out", type=str, default="output/correlation", help = "output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    if args.seed is not None:
        set_seed(args.seed)
    
    sence = "scenes/" + args.world + ".xml"
    model = "output/" + args.world + "/model_final.pt"

    evaluate_franka_correlation(model, sence, args.samples, args.out, device)