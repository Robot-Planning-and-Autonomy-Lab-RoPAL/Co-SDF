#!/usr/bin/env python3
"""
Render separate PNG images for RRT, NDF, and SDF trajectories
Each method gets its own figure showing workspace and configuration space
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
import matplotlib.patheffects as path_effects
import robot_tools


def shape_to_xy(shape, n=128):
    import polygon_tools as poly
    if isinstance(shape, poly.Polygon):
        return np.array([[p.x, p.y] for p in shape], dtype=float)
    if hasattr(shape, "to_polygon"):
        poly_approx = shape.to_polygon(n)
        return np.array([[p.x, p.y] for p in poly_approx], dtype=float)
    arr = np.asarray(shape, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
        return arr
    raise TypeError("Unsupported obstacle type for plotting.")


def plot_workspace_trajectory(ax, arm, path, color='red', label='Path', n_samples=6):
    """
    Clean Geometric Style: Uses exact robot thickness from robot_tools, 6 samples.
    """
    from matplotlib.patches import Polygon as PlotPolygon, Circle as PlotCircle
    import polygon_tools as poly
    
    # Sample 6 configurations (Start, 4 Shadows, Goal)
    indices = np.linspace(0, len(path) - 1, n_samples, dtype=int)
    
    # Pre-calculating EE path for the dashed line
    ee_positions = []
    for cfg in path:
        arm.set_link_angles(cfg)
        spine = arm.get_spine_points()
        ee_positions.append([spine[0][-1], spine[1][-1]])
    ee_positions = np.array(ee_positions)
    
    # Draw geometric arm configurations
    for i, idx in enumerate(indices):
        arm.set_link_angles(path[idx])
        spine = arm.get_spine_points()
        shapes = arm.get_collision_shapes() # Exact Circles/Polygons from robot_tools
        
        # Simple Color Map
        if i == 0:
            link_c, a, z = '#27AE60', 0.8, 10      # Premium Green Start
        elif i == len(indices) - 1:
            link_c, a, z = '#F39C12', 0.95, 11     # Premium Orange Target
        else:
            link_c, a, z = '#BDC3C7', 0.3, 5      # Premium Gray Shadows
            
        # Draw each part of the robot with correct scale
        for shape in shapes:
            if isinstance(shape, poly.Circle):
                # Check if it's the End-Effector (last spine point)
                # Using small epsilon for float comparison
                is_ee = (abs(shape.center.x - spine[0][-1]) < 1e-5 and 
                         abs(shape.center.y - spine[1][-1]) < 1e-5)
                c = 'black' if is_ee else '#E74C3C'
                patch = PlotCircle((shape.center.x, shape.center.y), shape.radius, 
                                  color=c, alpha=a, zorder=z+1, linewidth=0)
            else:
                # Link (Polygon)
                patch = PlotPolygon(shape_to_xy(shape), color=link_c, 
                                   alpha=a, zorder=z, linewidth=0)
            ax.add_patch(patch)
    
    # Workspace pathway (Subtle red dashed line)
    ax.plot(ee_positions[:, 0], ee_positions[:, 1], 'r--', 
            linewidth=1.2, alpha=0.5, zorder=1)
    
    return ee_positions


def plot_config_space_trajectory(ax, path, color='red', label='Path', alpha=0.8):
    """Plot trajectory in configuration space"""
    # Red dashed line for trajectory
    ax.plot(path[:, 0], path[:, 1], 'r--', linewidth=2, alpha=0.8)
    
    # Mark start and end with simple red circles
    ax.plot(path[0, 0], path[0, 1], 'ro', markersize=8, zorder=20)
    ax.plot(path[-1, 0], path[-1, 1], 'ro', markersize=8, zorder=20)


def setup_axes(ax, obstacles, cspace_array, xlim, ylim, theta1_lim, theta2_lim, col_map, title_prefix):
    """Setup workspace and configuration space axes (arm_config_space style)"""
    # Plot obstacles in workspace with shadow effect
    h_obs = []
    for o in obstacles:
        xy = shape_to_xy(o, n=256)
        h_obs.append(PlotPolygon(xy, closed=True, zorder=1))
    
    c_obs = PatchCollection(h_obs, cmap=col_map, edgecolors='none')
    c_obs.set_array(np.linspace(0, 1.0, len(obstacles) + 1)[1:])
    c_obs.set_path_effects([path_effects.withSimplePatchShadow(
        offset=(1, -1), shadow_rgbFace='gray', alpha=0.3)])
    ax[0].add_collection(c_obs)
    
    # Workspace settings
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    ax[0].set_aspect('equal')
    ax[0].xaxis.set_major_locator(MultipleLocator(2.0))
    ax[0].xaxis.set_minor_locator(MultipleLocator(1.0))
    ax[0].yaxis.set_major_locator(MultipleLocator(2.0))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1.0))
    ax[0].grid(which='major', axis='both', linestyle='--', alpha=0.3)
    ax[0].set_axisbelow(True)
    ax[0].minorticks_off()
    
    # Professional styling for workspace
    for spine in ax[0].spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#34495E')
        spine.set_alpha(0.8)
    for label in ax[0].get_xticklabels() + ax[0].get_yticklabels():
        label.set_color('#2C3E50')
        label.set_alpha(0.8)
    ax[0].tick_params(colors='gray', length=0, width=0)
    
    # Configuration space settings  
    ax[1].set_xlim(theta1_lim[0], theta1_lim[-1])
    ax[1].set_ylim(theta2_lim[0], theta2_lim[-1])
    ax[1].set_aspect('equal')
    ax[1].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax[1].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax[1].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax[1].set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax[1].grid(which='major', axis='both', linestyle='--', alpha=0.3)
    ax[1].set_axisbelow(True)
    ax[1].minorticks_off()
    
    # Professional styling for config space
    for spine in ax[1].spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#34495E')
        spine.set_alpha(0.8)
    ax[1].xaxis.label.set_color('#2C3E50')
    ax[1].yaxis.label.set_color('#2C3E50')
    for label in ax[1].get_xticklabels() + ax[1].get_yticklabels():
        label.set_color('#2C3E50')
        label.set_alpha(0.8)
    ax[1].tick_params(colors='gray', length=0, width=0)
    
    # Plot C-space obstacles
    cspace_masked = np.ma.masked_where(cspace_array == 0.0, cspace_array)
    col_map.set_bad(color='white')
    ax[1].imshow(cspace_masked.T, origin='lower', cmap=col_map, alpha=0.95,
                 extent=[theta1_lim[0], theta1_lim[-1], theta2_lim[0], theta2_lim[-1]],
                 interpolation='bilinear')


def render_single_trajectory_png(path, world_file, output_file, method_name, nx, n_samples=15):
    """
    Render PNG for a single trajectory
    
    Args:
        path: trajectory array
        world_file: YAML world configuration file
        output_file: output PNG filename
        method_name: name of the method (e.g., 'RRT', 'NDF', 'SDF')
        nx: configuration space resolution
        n_samples: number of arm configurations to show
    """
    print(f"Loading world from {world_file}...")
    arm_problem = robot_tools.PlanningProblem(world_file)
    theta, c_space = arm_problem.construct_config_space(nx)
    t1_min, t1_max = float(theta[0][0]), float(theta[0][-1])
    t2_min, t2_max = float(theta[1][0]), float(theta[1][-1])
    
    try:
        map_lims = arm_problem.workspace.limits
    except KeyError:
        map_lims = [[0, 10], [0, 10]]
    
    # Create combined figure
    fig, ax = plt.subplots(1, 2, figsize=(16, 7))
    
    # Use single blue color for obstacles (matching arm_config_space.py)
    import matplotlib.colors
    col_map = matplotlib.colors.ListedColormap([[0.20, 0.50, 0.70, 1.0]])
    
    # Setup axes
    setup_axes(ax, arm_problem.workspace.obstacles, c_space, 
               map_lims[0], map_lims[1], [t1_min, t1_max], [t2_min, t2_max], 
               col_map, method_name)
    
    # Define color for the method
    colors = {
        'RRT': 'tab:blue',
        'NDF': 'tab:green',
        'SDF': 'tab:orange'
    }
    color = colors.get(method_name, 'tab:purple')
    
    print(f"Plotting {method_name} trajectory with {len(path)} waypoints...")
    
    # Plot in workspace
    plot_workspace_trajectory(ax[0], arm_problem.robot, path, 
                             color=color, label=method_name, n_samples=n_samples)
    
    # Plot in configuration space
    plot_config_space_trajectory(ax[1], path, color=color, label=method_name)
    
    plt.tight_layout()
    
    # Save combined
    print(f"Saving combined PNG to {output_file}...")
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    
    # NEW: Save individual figures for paper layout (6 images total)
    base_path = os.path.splitext(output_file)[0]
    ws_file = f"{base_path}_ws.png"
    cs_file = f"{base_path}_cs.png"
    
    # Save Workspace only
    extent0 = ax[0].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(ws_file, dpi=300, bbox_inches=extent0)
    
    # Save C-Space only
    extent1 = ax[1].get_tightbbox(fig.canvas.get_renderer()).transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(cs_file, dpi=300, bbox_inches=extent1)
    
    plt.close(fig)
    print(f"{method_name} individual plots saved to {ws_file} and {cs_file}.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D path PNG renderer (separate images)")
    parser.add_argument("--world", type=str, default="obs10", help="YAML world file")
    parser.add_argument("--png-nx", type=int, default=201, help="Config-space resolution")
    parser.add_argument("--n-samples", type=int, default=6, help="Number of arm configs to show")
    args = parser.parse_args()
    
    # Define file paths
    world_file = f"config/{args.world}.yaml"
    
    methods = {
        'RRT': f"output/traj/{args.world}/rrt.npy",
        'NDF': f"output/traj/{args.world}/ndf.npy",
        'SDF': f"output/traj/{args.world}/sdf.npy"
    }
    
    output_files = {
        'RRT': f"output/traj/{args.world}/rrt_trajectory.png",
        'NDF': f"output/traj/{args.world}/ndf_trajectory.png",
        'SDF': f"output/traj/{args.world}/sdf_trajectory.png"
    }
    
    # Create output directory if needed
    os.makedirs(f"output/traj/{args.world}", exist_ok=True)
    
    # Render each trajectory separately
    for method_name, path_file in methods.items():
        if os.path.exists(path_file):
            print(f"Loading {method_name} path from {path_file}...")
            path = np.load(path_file)
            render_single_trajectory_png(path, world_file, output_files[method_name], 
                                        method_name, nx=args.png_nx, 
                                        n_samples=args.n_samples)
        else:
            print(f"Warning: {path_file} not found, skipping {method_name}")
    
    print("All trajectory PNGs generated successfully!")
