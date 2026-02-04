#!/usr/bin/env python3
import numpy as np
import os
import time
import random
from ndf_casadi import CasADiPlanner
from trajecor_verify import TrajectoryEvaluator

class LocalShortcutOptimizer:
    def __init__(self, model_path, scene_xml, device="cuda"):
        self.planner = CasADiPlanner(model_path, scene_xml, device=device)
        self.evaluator = TrajectoryEvaluator(scene_xml, model=self.planner.opt_base.model)
        
    def optimize(self, initial_path, iterations=20, max_segment_len=25, n_local_points=15,
                 solve_margin=0.02, verify_margin=0.03, target_pts=100):
        """
        Iteratively pick segments and repair them using IPOPT.
        """
        # 0) Precompile (one-time) - Not counted in loop/solve time
        print(f"Pre-compiling {n_local_points}-waypoint model...")
        t_compile0 = time.time()
        _ = self.planner.get_l4c_model(n_local_points)
        compile_time = time.time() - t_compile0
        
        current_path = initial_path.copy()
        print(f"Starting Local Shortcut Repair: {iterations} iterations on {self.planner.device}...")
        
        t_loop0 = time.time()
        n_success = 0
        
        for k in range(iterations):
            n_pts = len(current_path)
            if n_pts < 5: break
            
            # 1. Randomly pick a segment
            i = random.randint(0, n_pts - 3)
            j = random.randint(i + 2, min(i + max_segment_len, n_pts - 1))
            
            p_start = current_path[i]
            p_end = current_path[j]
            
            # 2. Solve small local NLP with very tight iteration limit
            # Lower solve_margin is faster and more likely to succeed
            success, local_path, _, stats = self.planner.solve(
                p_start, p_end, 
                formulation='soft', # Use soft penalty by default for speed
                solver='ipopt',
                n_waypoints=n_local_points,
                verbose=False,
                safety_margin=solve_margin,
                max_iter=50
            )
            
            if success:
                # 3. Verify safety of the NEW segment (more strictly)
                metrics = self.evaluator.evaluate(local_path, safety_margin=verify_margin)
                if metrics['success']:
                    # 4. Local Acceptance Logic: Length and jumping avoidance
                    old_seg = current_path[i:j+1]
                    old_len = np.sum(np.linalg.norm(old_seg[1:] - old_seg[:-1], axis=1))
                    new_len = np.sum(np.linalg.norm(local_path[1:] - local_path[:-1], axis=1))
                    
                    # Accept if length doesn't increase by more than 5%
                    if new_len <= 1.05 * old_len:
                        new_path = np.vstack([
                            current_path[:i],
                            local_path,
                            current_path[j+1:]
                        ])
                        current_path = new_path
                        n_success += 1
        
        loop_time = time.time() - t_loop0
        current_path_len = np.sum(np.linalg.norm(current_path[1:] - current_path[:-1], axis=1))
        print(f"Repair Loop Finished: {n_success}/{iterations} successful shortcuts in {loop_time:.2f}s")
        
        # --- Final Step: Global Resampling for Uniform Velocity ---
        t_post0 = time.time()
        # Instead of fixed 100 pts, we use a fixed "density" so shorter paths look faster
        # RRT 100 pts for ~10.0 length means roughly 10 pts per unit length
        target_pts = max(30, int(current_path_len * 10)) 
        print(f"Finalizing trajectory with distance-based resampling ({target_pts} pts)...")
        final_path = self.resample_path(current_path, target_pts=target_pts)
        
        # Final safety check after interpolation
        final_metrics = self.evaluator.evaluate(final_path, safety_margin=verify_margin)
        if not final_metrics["success"]:
            print("  Warning: Resampling introduced collisions. Reverting to pre-resampled path.")
            final_path = current_path
            
        post_time = time.time() - t_post0
        total_time = loop_time + post_time # Exclude compile_time per user request
        
        print(f"Timing Details (excluding pre-compilation):")
        print(f"  Loop:    {loop_time:.2f}s")
        print(f"  Post:    {post_time:.2f}s")
        print(f"  TOTAL:   {total_time:.2f}s (Compile: {compile_time:.2f}s hidden)")
        
        return final_path, total_time

    def resample_path(self, path, target_pts=100):
        """
        Resample the path to have uniform distance between waypoints.
        """
        # 1. Calculate cumulative distance along path
        diffs = np.linalg.norm(path[1:] - path[:-1], axis=1)
        cum_dist = np.concatenate(([0], np.cumsum(diffs)))
        total_dist = cum_dist[-1]
        
        # 2. Generate target distances
        target_dist = np.linspace(0, total_dist, target_pts)
        
        # 3. Interpolate each joint separately
        new_path = np.zeros((target_pts, path.shape[1]))
        for i in range(path.shape[1]):
            new_path[:, i] = np.interp(target_dist, cum_dist, path[:, i])
            
        return new_path

if __name__ == "__main__":
    import argparse
    from rrt import RRT
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scenes/table_obs10.xml")
    parser.add_argument("--model", type=str, default="output/table_obs10/model_final.pt")
    parser.add_argument("--iter", type=int, default=20)
    args = parser.parse_args()
    
    optimizer = LocalShortcutOptimizer(args.model, args.scene)
    rrt_planner = RRT(args.scene)
    evaluator = TrajectoryEvaluator(args.scene, model=optimizer.planner.opt_base.model)
    
    # 1. Get raw RRT path
    print("Generating RRT path...")
    path_raw = rrt_planner.plan(output_file=None, timeout=10.0)
    if path_raw is None:
        print("RRT failed. Exit.")
        sys.exit(1)
    
    m_raw = evaluator.evaluate(path_raw)
    print(f"Initial RRT: {len(path_raw)} pts, Length={m_raw['path_length']:.4f}, Clearance={m_raw['min_clearance']:.4f}")
    
    # 2. Run Local Shortcut Repair
    path_opt, solve_time = optimizer.optimize(path_raw, iterations=args.iter)
    
    # 3. Final Evaluate
    m_opt = evaluator.evaluate(path_opt, solve_time=solve_time)
    print("\n" + "="*40)
    print("LOCAL SHORTCUT RESULTS")
    print("="*40)
    print(f"Solve Time:    {solve_time:.4f}s")
    print(f"Path Pts:      {len(path_raw)} -> {len(path_opt)}")
    print(f"Path Length:   {m_raw['path_length']:.4f} -> {m_opt['path_length']:.4f}")
    print(f"Min Clearance: {m_raw['min_clearance']:.4f} -> {m_opt['min_clearance']:.4f}")
    print(f"Success:       {m_opt['success']}")
    
    os.makedirs("output/traj", exist_ok=True)
    np.save("output/traj/rrt.npy", path_raw)
    np.save("output/traj/opt.npy", path_opt)
    print(f"\nSaved to output/traj/ (rrt.npy and opt.npy)")
