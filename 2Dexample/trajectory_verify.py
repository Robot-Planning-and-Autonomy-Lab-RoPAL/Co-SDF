#!/usr/bin/env python3
import argparse
import numpy as np
import robot_tools

def check_path_collision(path: np.ndarray, problem: robot_tools.PlanningProblem, 
                        interp_steps: int = 0, verbose: bool = True) -> tuple:
    """
    Check path for collisions with optional interpolation.
    
    Args:
        path: (T, 2) array of joint configurations
        problem: PlanningProblem with robot and workspace
        interp_steps: Number of interpolation steps between waypoints
        verbose: Whether to print collision details
    
    Returns:
        (is_collision_free, num_collisions)
    """
    def in_collision(q):
        problem.robot.set_link_angles(q)
        fp = problem.robot.get_current_polygon()
        return any(fp.intersect(o) for o in problem.workspace.obstacles)

    n_collisions = 0

    # Check waypoints
    for i, q in enumerate(path):
        if in_collision(q):
            if verbose:
                print(f"  Collision at waypoint {i}")
            n_collisions += 1

    # Check interpolated points
    if interp_steps > 0 and len(path) > 1:
        for i in range(len(path) - 1):
            q1 = path[i]
            q2 = path[i + 1]
            for s in range(1, interp_steps + 1):
                q = q1 + (q2 - q1) * (s / (interp_steps + 1))
                q = np.mod(q, 2 * np.pi)
                if in_collision(q):
                    if verbose:
                        print(f"  Collision along segment {i}-{i+1} at substep {s}")
                    n_collisions += 1

    is_collision_free = (n_collisions == 0)
    if verbose:
        if is_collision_free:
            print("SUCCESS: Path is collision-free.")
        else:
            print(f"FAILURE: Found {n_collisions} collision states.")
    
    return is_collision_free, n_collisions


def verify_trajectory(path_file: str, world_file: str, interp_steps: int = 0) -> bool:
    """Load path and world from files, then check for collisions."""
    print(f"Loading path from {path_file}...")
    path = np.load(path_file)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError("Expected path of shape (T, 2) for 2D joints")

    print(f"Loading world from {world_file}...")
    problem = robot_tools.PlanningProblem(world_file)

    is_collision_free, _ = check_path_collision(path, problem, interp_steps, verbose=True)
    return is_collision_free

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D path collision verifier")
    parser.add_argument("--path", default="output/traj/optim.npy", help="Path .npy file (T,2)")
    parser.add_argument("--world", default="config/block_world5.yaml", help="YAML world file")
    parser.add_argument("--interp", type=int, default=10, help="Interpolation steps per segment")
    args = parser.parse_args()

    ok = verify_trajectory(args.path, args.world, args.interp)
    exit(0 if ok else 1)