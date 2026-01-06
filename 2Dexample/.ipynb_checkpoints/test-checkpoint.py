import torch
import torch.nn as nn
import numpy as np
import os
import matplotlib.pyplot as plt
from common.models import select_model
import robot_tools

class PeriodicEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)

def build_model(device):
    embedding = PeriodicEmbedding()
    model_core = select_model("sll", 4, 20, 128)
    model = nn.Sequential(embedding, model_core).to(device)
    return model

def load_and_eval(model_path, X_test, Y_test, device):
    model = build_model(device)
    data = torch.load(model_path, map_location=device)
    
    if isinstance(data, dict) and "state_dict" in data:
        state_dict = data["state_dict"]
    else:
        state_dict = data
        
    # Mapping keys if necessary
    model_keys = set(model.state_dict().keys())
    sd_keys = set(state_dict.keys())
    
    if sd_keys == model_keys:
        model.load_state_dict(state_dict)
    else:
        # Try to fix "1.1.weight" -> "1.weight" etc or vice versa
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if len(missing) > 0 or len(unexpected) > 0:
            if len(missing) > 0 and len(unexpected) > 0:
                new_sd = {}
                for uk in unexpected:
                    val = state_dict[uk]
                    if uk.startswith("1."):
                        nk = uk[2:]
                        if nk in missing:
                            new_sd[nk] = val
                            continue
                    nk = "1." + uk
                    if nk in missing:
                        new_sd[nk] = val
                        continue
                if new_sd:
                    model.load_state_dict(new_sd, strict=False)
    
    model.eval()
    with torch.no_grad():
        X_test_t = torch.Tensor(X_test).to(device)
        preds = model(X_test_t).cpu().numpy().flatten()
        
    y_pred = np.ones_like(preds)
    y_pred[preds < 0] = -1
    
    accuracy = (y_pred == Y_test).mean()
    mse = ((preds - Y_test)**2).mean()
    
    # Calculate IoU for collision region (-1)
    gt_collision = (Y_test == -1)
    pred_collision = (y_pred == -1)
    intersection = np.logical_and(gt_collision, pred_collision).sum()
    union = np.logical_or(gt_collision, pred_collision).sum()
    iou = intersection / union if union > 0 else 1.0
    
    return accuracy, mse, iou, preds

def plot_decision_boundary(model, model_name, X_test, Y_test, world_file, device):
    print(f"Plotting decision boundary for {model_name}...")
    arm_problem = robot_tools.PlanningProblem(world_file)
    
    # Create grid for contour
    res = 256
    x = np.linspace(0, 2*np.pi, res)
    y = np.linspace(0, 2*np.pi, res)
    X, Y = np.meshgrid(x, y)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)
    
    model.eval()
    with torch.no_grad():
        grid_points_t = torch.Tensor(grid_points).to(device)
        Z = model(grid_points_t).cpu().numpy().reshape(res, res)
    
    plt.figure(figsize=(10, 8))
    
    # 0-level contour (decision boundary)
    cs = plt.contour(X, Y, Z, levels=[0], colors='black', linewidths=2)
    plt.clabel(cs, inline=1, fontsize=10, fmt={0: 'f(q)=0'})
    
    # Plot GT points
    xi = X_test[Y_test == -1]
    xo = X_test[Y_test == 1]
    
    # Subsample points for clearer visualization if too many
    max_pts = 2000
    if len(xi) > max_pts:
        idx = np.random.choice(len(xi), max_pts, replace=False)
        xi = xi[idx]
    if len(xo) > max_pts:
        idx = np.random.choice(len(xo), max_pts, replace=False)
        xo = xo[idx]

    plt.scatter(xo[:,0], xo[:,1], c='blue', s=2, label="GT Free", alpha=0.3)
    plt.scatter(xi[:,0], xi[:,1], c='red', s=2, label="GT Collision", alpha=0.3)
    
    pi = np.pi
    plt.xticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=12)
    plt.yticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'], fontsize=12)
    plt.xlabel(r"$\theta_1$ (rad)", fontsize=14)
    plt.ylabel(r"$\theta_2$ (rad)", fontsize=14)
    plt.title(f"Decision Boundary: {model_name}", fontsize=16)
    plt.legend(loc='upper right', markerscale=5)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/boundary_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_output_distribution(preds, Y_test, model_name, epsilon=0.5):
    print(f"Plotting output distribution for {model_name}...")
    
    # Select points near boundary
    near_idx = np.abs(preds) < epsilon
    near_preds = preds[near_idx]
    near_labels = Y_test[near_idx]
    
    plt.figure(figsize=(10, 6))
    
    # Collision points (GT = -1)
    col_preds = near_preds[near_labels == -1]
    # Free points (GT = 1)
    free_preds = near_preds[near_labels == 1]
    
    plt.hist(free_preds, bins=50, alpha=0.6, label='GT Free', color='blue', density=True)
    plt.hist(col_preds, bins=50, alpha=0.6, label='GT Collision', color='red', density=True)
    
    plt.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Decision Boundary')
    plt.xlabel(r"$f_\theta(q)$ (Model Output)", fontsize=14)
    plt.ylabel("Density", fontsize=14)
    plt.title(f"Output Distribution Near Boundary (|f(q)| < {epsilon}): {model_name}", fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    out_dir = "plots"
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/distribution_{model_name}.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_random_test_set(world_file, n_points=3000):
    print(f"Generating random test set with {n_points} points for world: {world_file}")
    arm_problem = robot_tools.PlanningProblem(world_file)
    X_test = np.random.uniform(0, 2 * np.pi, (n_points, 2))
    Y_test = np.zeros(n_points)
    for i in range(n_points):
        arm_problem.robot.set_link_angles(X_test[i])
        fp = arm_problem.robot.get_current_polygon()
        in_collision = any(fp.intersect(o) for o in arm_problem.workspace.obstacles)
        Y_test[i] = -1 if in_collision else 1
    return X_test, Y_test

def generate_boundary_test_set(world_file, n_points=5000, sigma=0.03):
    print(f"Generating boundary-focused test set ({n_points} points)...")
    arm_problem = robot_tools.PlanningProblem(world_file)
    
    lo = np.array([0, 0])
    hi = np.array([2*np.pi, 2*np.pi])
    span = hi - lo
    
    X_boundary = []
    Y_boundary = []
    
    while len(X_boundary) < n_points:
        # Sample points and find ones near boundary using the straddling technique
        q1 = np.random.uniform(0, 2*np.pi, (n_points, 2))
        noise = np.random.randn(n_points, 2) * (sigma * span)
        q2 = np.clip(q1 + noise, lo, hi)
        
        # Labels
        l1 = []
        l2 = []
        for i in range(n_points):
            # Q1
            arm_problem.robot.set_link_angles(q1[i])
            fp1 = arm_problem.robot.get_current_polygon()
            c1 = any(fp1.intersect(o) for o in arm_problem.workspace.obstacles)
            l1.append(-1 if c1 else 1)
            
            # Q2
            arm_problem.robot.set_link_angles(q2[i])
            fp2 = arm_problem.robot.get_current_polygon()
            c2 = any(fp2.intersect(o) for o in arm_problem.workspace.obstacles)
            l2.append(-1 if c2 else 1)
            
        l1 = np.array(l1)
        l2 = np.array(l2)
        
        # Boundary points are those where labels differ
        mask = (l1 != l2)
        X_boundary.extend(q1[mask])
        Y_boundary.extend(l1[mask])
        X_boundary.extend(q2[mask])
        Y_boundary.extend(l2[mask])
        
    X_boundary = np.array(X_boundary)[:n_points]
    Y_boundary = np.array(Y_boundary)[:n_points]
    
    return X_boundary, Y_boundary

def plot_test_points(X_test, Y_test, world_file):
    print("Plotting test points...")
    arm_problem = robot_tools.PlanningProblem(world_file)
    theta, c_space = arm_problem.construct_config_space(nx=101)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(c_space.T, origin="lower",
              extent=[theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]],
              cmap="gray_r", alpha=0.3)
    
    xi = X_test[Y_test == -1]
    xo = X_test[Y_test == 1]
    
    if len(xi):
        plt.scatter(xi[:,0], xi[:,1], c='red', s=1, label="Collision", alpha=0.5)
    if len(xo):
        plt.scatter(xo[:,0], xo[:,1], c='blue', s=1, label="Free", alpha=0.5)
        
    plt.xlabel(r"$\theta_1$")
    plt.ylabel(r"$\theta_2$")
    plt.title(f"Randomly Sampled Test Points ({len(X_test)})")
    plt.legend()
    
    pi = np.pi
    plt.xticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    plt.yticks([0, pi/2, pi, 3*pi/2, 2*pi], [r'0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    out_path = "test_points.png"
    plt.savefig(out_path, dpi=300)
    print(f"Plot saved to {out_path}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Set paths relative to script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    world_file = os.path.join(script_dir, 'config/block_world.yaml')
    n_points = 10000
    n_trials = 5
    
    # Evaluate models
    models_to_test = [
        os.path.join(script_dir, "output/uniform/model_final.pt"),
        os.path.join(script_dir, "output/uniform_aabb/model_final.pt"),
        os.path.join(script_dir, "output/gaussian/model_final.pt"),
        os.path.join(script_dir, "output/gaussian_aabb/model_final.pt")
    ]
    
    # Storage: {model_path: {"acc": [], "mse": [], "iou": [], "b_acc": []}}
    all_results = {m: {"acc": [], "mse": [], "iou": [], "b_acc": []} for m in models_to_test}

    for trial in range(n_trials):
        print(f"\n" + "="*40)
        print(f"Trial {trial + 1} / {n_trials}")
        print("="*40)
        
        # Test sets
        X_test, Y_test = generate_random_test_set(world_file, n_points)
        X_bound, Y_bound = generate_boundary_test_set(world_file, n_points // 2)
        
        for model_path in models_to_test:
            if not os.path.exists(model_path): continue
            name = os.path.basename(os.path.dirname(model_path))
            
            # 1. Global Metrics
            acc, mse, s_iou, preds = load_and_eval(model_path, X_test, Y_test, device)
            
            # 2. Boundary Accuracy
            b_acc, _, _, _ = load_and_eval(model_path, X_bound, Y_bound, device)
            
            all_results[model_path]["acc"].append(acc)
            all_results[model_path]["mse"].append(mse)
            all_results[model_path]["iou"].append(s_iou)
            all_results[model_path]["b_acc"].append(b_acc)
            
            print(f"[{name}] Global Acc: {acc*100:.2f}%, Boundary Acc: {b_acc*100:.2f}%, IoU: {s_iou:.4f}")

    # Final Summary
    print("\n" + "#"*40)
    print(f" FINAL RESULTS (Averaged over {n_trials} trials) ")
    print("#"*40)
    
    for model_path in models_to_test:
        if not os.path.exists(model_path): continue
        name = os.path.basename(os.path.dirname(model_path))
        r = all_results[model_path]
        
        print(f"\nModel: {name}")
        print(f"  Accuracy (Global):   {np.mean(r['acc'])*100:.2f}% ± {np.std(r['acc'])*100:.2f}%")
        print(f"  Accuracy (Boundary): {np.mean(r['b_acc'])*100:.2f}% ± {np.std(r['b_acc'])*100:.2f}%")
        print(f"  IoU (Sampled):       {np.mean(r['iou']):.4f} ± {np.std(r['iou']):.4f}")
        print(f"  MSE (Global):        {np.mean(r['mse']):.4f} ± {np.std(r['mse']):.4f}")
