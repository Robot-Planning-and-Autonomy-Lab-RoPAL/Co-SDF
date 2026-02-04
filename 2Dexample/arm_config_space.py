import numpy as np
import matplotlib.pyplot as plt
import polygon_tools as poly
import robot_tools
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors
from matplotlib.colors import TwoSlopeNorm
import argparse
import shutil
from matplotlib.ticker import MultipleLocator
from cspace_sdf import cobs_sdf_on_torus
import matplotlib.patheffects as path_effects
import torch

""" 

Plot the config space from Introduction to Autonomous Mobile Robots Fig 6.1 

Requires: numpy, matplotlib, argparse

Author: Nicholas Lawrance (nicholas.lawrance@mavt.ethz.ch)

"""

def shape_to_xy(shape, n=128):

    if isinstance(shape, poly.Polygon):
      return np.array([[p.x, p.y] for p in shape], dtype=float)
    if hasattr(shape, "to_polygon"):
      poly_approx = shape.to_polygon(n)
      return np.array([[p.x, p.y] for p in poly_approx], dtype=float)
    arr = np.asarray(shape, dtype=float)
    if arr.ndim == 2 and arr.shape[1] == 2:
      return arr
    raise TypeError("Unsupported obstacle type for plotting; need Polygon, Circle(with to_polygon), or Nx2 array.")
  
if shutil.which('latex'):
    plt.rc('font', **{'family': 'serif', 'serif': ['Times', 'Computer Modern Roman']})
    plt.rc('text', usetex=True)
else:
    plt.rc('font', family='serif')

# IROS-style plotting settings
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

# Use a more premium, darker color for obstacles
colour_map = matplotlib.colors.ListedColormap([[0.20, 0.50, 0.70, 1.0]])  # Rich Steel Blue


def angle_wrap(angles):
    return angles % (2 * np.pi)


def linear_path(points, n=50):
    c_point = points[0]
    path = np.array([c_point])

    for end_point in points[1:]:
        new_path = np.linspace(path[-1], end_point, n)       # Requires numpy > 1.16.0
        path = np.concatenate((path, new_path[1:]))

    return path

def plot_config_space(ax, 
                      obstacles, 
                      arm, 
                      cspace_array, 
                      col_map,
                      xlim, 
                      ylim, 
                      theta1_lim, 
                      theta2_lim,
                      heat=None, 
                      heat_extent=None, 
                      heat_cmap=None, 
                      heat_norm=None):
    
    h_obs = []
    for o in obstacles:
      xy = shape_to_xy(o, n=256)
      h_obs.append(PlotPolygon(xy, closed=True, zorder=1))
        
    c_obs = PatchCollection(h_obs, cmap=col_map, edgecolors='none')
    # This sets colors for some reason (command in Polygon does not)
    c_obs.set_array(np.linspace(0, 1.0, len(obstacles) + 1)[1:])
    # Add shadow effect to obstacles
    c_obs.set_path_effects([path_effects.withSimplePatchShadow(offset=(1, -1), shadow_rgbFace='gray', alpha=0.3)])
    ax[0].add_collection(c_obs)

    # Plot arm links with a nice navy blue color
    spine_points = arm.get_spine_points()
    h_arm, = ax[0].plot(*spine_points, c='#34495E', lw=5.0, solid_capstyle='round',
                        path_effects=[path_effects.SimpleLineShadow(offset=(0.6, -0.6), alpha=0.3), 
                                     path_effects.Normal()])
    
    # Plot joints (all points except the last one which is end effector)
    h_joints, = ax[0].plot(spine_points[0][:-1], spine_points[1][:-1], 'o', 
                           color='#E74C3C', markersize=11, zorder=5, 
                           markeredgecolor='white', markeredgewidth=1.5)
    
    # Plot end effector with different color
    h_ee, = ax[0].plot(spine_points[0][-1:], spine_points[1][-1:], 'o', 
                       color='#F39C12', markersize=12, zorder=5,
                       markeredgecolor='white', markeredgewidth=1.5)

    for a in ax:
        a.set_aspect('equal')

    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    # Only show even numbers on workspace axes
    ax[0].set_xticks([2, 4, 6, 8, 10])
    ax[0].set_yticks([2, 4, 6, 8, 10])
    # Make workspace tick labels darker
    for label in ax[0].get_xticklabels() + ax[0].get_yticklabels():
        label.set_color('#2C3E50')
        label.set_alpha(0.8)

    ax[1].set_xlabel(r'$\theta_1$')
    ax[1].set_ylabel(r'$\theta_2$')
    ax[1].set_xlim(theta1_lim[0], theta1_lim[-1])
    ax[1].set_ylim(theta2_lim[0], theta2_lim[-1])

    # This is a bit dumb, should probably just assume [0, 2pi) everywhere, but meh
    ax[0].xaxis.set_major_locator(MultipleLocator(2.0))
    ax[0].xaxis.set_minor_locator(MultipleLocator(1.0))
    ax[0].yaxis.set_major_locator(MultipleLocator(2.0))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1.0))
    ax[1].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax[1].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax[1].xaxis.set_minor_locator(MultipleLocator(np.pi/4))
    ax[1].yaxis.set_minor_locator(MultipleLocator(np.pi/4))

    ax[1].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    ax[1].set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
    
    # Make cspace axis tick labels darker
    for label in ax[1].get_xticklabels():
        label.set_color('#2C3E50')
        label.set_alpha(0.8)
    for label in ax[1].get_yticklabels():
        label.set_color('#2C3E50')
        label.set_alpha(0.8)
    
    for a in ax:
        a.grid(which='major', axis='both', linestyle='--', alpha=0.3)
        a.set_axisbelow(True)
        # Hide minor ticks to avoid visual clutter
        a.minorticks_off()
        # Darker box and axes for better print quality
        for spine in a.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('#34495E')
            spine.set_alpha(0.8)
        # Darker axis ticks and labels
        a.xaxis.label.set_color('#2C3E50')
        a.xaxis.label.set_alpha(0.9)
        a.yaxis.label.set_color('#2C3E50')
        a.yaxis.label.set_alpha(0.9)
        # Darker tick labels (numbers)
        for label in a.get_xticklabels():
            label.set_color('#2C3E50')
            label.set_alpha(0.8)
        for label in a.get_yticklabels():
            label.set_color('#2C3E50')
            label.set_alpha(0.8)
        # Hide tick marks to avoid visual clutter
        a.tick_params(colors='gray', length=0, width=0)

    cspace_array = np.ma.masked_where(cspace_array == 0.0, cspace_array)
    col_map.set_bad(color='white')
    # Add edge effect to configuration space obstacles
    ax[1].imshow(cspace_array.transpose(), origin='lower', cmap=col_map,
                 extent=[theta1_lim[0], theta1_lim[1], theta2_lim[0], theta2_lim[1]],
                 interpolation='bilinear', alpha=0.95)
    
    # SDF
    if heat is not None:
        hm = ax[1].imshow(heat.T, origin='lower', extent=heat_extent,
                          cmap=(heat_cmap or "RdBu_r"), norm=heat_norm)

        ax[1]._heatmap_handle = hm

    return h_arm, h_joints, h_ee

class ArmAnimator(object):
    h_arm = None
    plot_artists = []

    def __init__(self, 
                 arm, 
                 obstacles, 
                 cspace_array, 
                 path, 
                 x_lim, 
                 y_lim, 
                 t1_lim, 
                 t2_lim,
                 col_map=cm.viridis.copy(), 
                 shadow_skip=0):

        self.fig, self.ax = plt.subplots(1, 2)
        self.fig.set_size_inches([9.6, 5.4])  # 1920*1080 at 200 dpi
        self.arm = arm
        self.obstacles = obstacles
        self.cspace_array = cspace_array
        self.path = path
        self.cmap = col_map
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.t1lim = t1_lim
        self.t2lim = t2_lim
        self.max_frames = self.path.shape[0]
        self.end_effector_path = poly.PointList([])
        self._shadow_skip = shadow_skip

    def init_fig(self):
        for a in self.ax:
            a.cla()

        self.arm.set_link_angles(self.path[0])
        self.h_arm, self.h_joints, self.h_ee = plot_config_space(self.ax, self.obstacles, self.arm, self.cspace_array, self.cmap, self.x_lim,
                                       self.y_lim, self.t1lim, self.t2lim)
                                                   
        self.last_break = 0
        self.h_path, = self.ax[1].plot(self.path[:1, 0], self.path[:1, 1], 'r--')
        self.h_pathend, = self.ax[1].plot(self.path[0, 0], self.path[0, 1], 'ro')

        self.end_effector_path = poly.PointList([self.arm.get_end_effector_position()])
        self.h_ee_path, = self.ax[0].plot([self.end_effector_path[0].x], [self.end_effector_path[0].y], 'r--')
        self.h_ee_pathend, = self.ax[0].plot([self.end_effector_path[0].x], [self.end_effector_path[0].y], 'ro')
        self.plot_artists = [self.h_arm, self.h_joints, self.h_ee, self.h_path, self.h_pathend, self.h_ee_path, self.h_ee_pathend]

        return self.plot_artists

    def animate(self, i):

        # If plotting extra arm shadows, add them to the plot_artists
        if self._shadow_skip != 0 and i % self._shadow_skip == 0:
            gv = 0.9-float(i)/self.max_frames*0.9
            h_arm_shadow = self.ax[0].plot(*self.arm.get_spine_points(), c=[gv, gv, gv], lw=1.5)
            h_arm_shadow.extend(self.plot_artists)
            self.plot_artists = h_arm_shadow

        self.arm.set_link_angles(self.path[i])
        spine_points = self.arm.get_spine_points()
        self.h_arm.set_data(*spine_points)
        self.h_joints.set_data(spine_points[0][:-1], spine_points[1][:-1])
        self.h_ee.set_data(spine_points[0][-1:], spine_points[1][-1:])

        # If the path crosses one of the boundaries, break it and add a new path
        if any(abs(self.path[i] - self.path[i-1]) > np.pi):
            old_path, = self.ax[1].plot(self.path[self.last_break:i, 0], self.path[self.last_break:i, 1], 'r--')
            self.plot_artists.append(old_path)
            self.last_break = i

        self.h_path.set_data(self.path[self.last_break:(i+1), 0], self.path[self.last_break:(i+1), 1])
        self.h_pathend.set_data([self.path[i, 0]], [self.path[i, 1]])

        self.end_effector_path.append(self.arm.get_end_effector_position())
        self.h_ee_path.set_data(*self.end_effector_path.get_xy())
        self.h_ee_pathend.set_data([self.end_effector_path[-1].x], [self.end_effector_path[-1].y])

        return self.plot_artists

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot the config space from Fig 6.1 in Intro to AMR textbook')
    parser.add_argument('-nx', type=int, default=201, help='Resolution (n points in each dimension)')
    parser.add_argument('-w', '--world', default='config/block_world1.yaml', help='World definition (obstacles)')
    parser.add_argument('-sa', '--save-animation', default=None, help='Save animation to file')
    parser.add_argument('-na', '--no-animation', action='store_true', help='Don\'t animate')
    parser.add_argument('--arm-shadows', type=int, default=0,
                        help='Plot shadows of arm position every n steps (0 for off)')
    args = parser.parse_args()


    arm_problem = robot_tools.PlanningProblem(args.world)
    theta, c_space = arm_problem.construct_config_space(args.nx)
    t1_min, t1_max = float(theta[0][0]), float(theta[0][-1])
    t2_min, t2_max = float(theta[1][0]), float(theta[1][-1])
    
    # Example path from textbook
    path_fit = np.polyfit([0.3, 1.6, 4.3, 5.9], [1.2, 0.8, 3.3, 3.2], 3)
    path_theta2 = np.linspace(0.3, 5.9, 300)
    path_theta1 = np.polyval(path_fit, path_theta2)
    path_full = angle_wrap(np.array([path_theta1, path_theta2]).T)

    ee_path = arm_problem.robot.end_effector_path(path_full)

    # Use subplot_mosaic with constrained layout for perfect box alignment
    mosaic = [['workspace', 'cspace'],
              ['workspace', 'ndf']]
    f1, ax_dict = plt.subplot_mosaic(mosaic, figsize=(12, 8), 
                                    width_ratios=[2.2, 1.0], layout='constrained')
    a1 = [ax_dict['workspace'], ax_dict['cspace'], ax_dict['ndf']]
    
    for ax in a1:
        ax.set_aspect('equal')

    arm_problem.robot.set_link_angles(path_full[0])
    try:
        map_lims = arm_problem.workspace.limits
    except KeyError:
        map_lims = [[0, 10], [0, 10]]
        
    # Build C-obstacle SDF
    # theta_per, sdf_grid = cobs_sdf_on_torus(theta, c_space)
    
    # vmax = np.nanpercentile(np.abs(sdf_grid), 99)
    # norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    # cmap = "seismic"
    
    # extent = [theta[0][0], theta[0][-1], theta[1][0], theta[1][-1]]
    
    # Set initial pose for better visualization
    initial_pose = np.array([np.pi/5, np.pi/2.5])  # ~36 degrees, ~72 degrees
    arm_problem.robot.set_link_angles(initial_pose)
    plot_config_space(
        a1,
        arm_problem.workspace.obstacles,
        arm_problem.robot,
        c_space,
        colour_map,
        map_lims[0],
        map_lims[1],
        theta[0][[0, -1]],
        theta[1][[0, -1]],
    )
    
    # Mark current configuration in configuration space
    a1[1].plot(initial_pose[0], initial_pose[1], 'o', color='#E74C3C', markersize=8, 
              markeredgewidth=1.5, markeredgecolor='white', zorder=10)
    
    # Hide redundant x-labels on the top C-space plot
    plt.setp(a1[1].get_xticklabels(), visible=False)
    a1[1].set_xlabel('')
    
    # Load NDF model and render SDF heatmap
    try:
        from common.models.gndf_2d import G_NDF_2D
        model_path = 'output/block_world1/model_final.pt'
        print(f"Loading NDF model from {model_path}")
        
        device = 'cpu'
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Extract parameters from meta if available
        if 'meta' in checkpoint:
            # meta: [n_layers, n_hidden, n_points_per_link, ...]
            n_layers, n_hidden, n_points_per_link = checkpoint['meta'][:3]
            print(f"Using model parameters from checkpoint: n_layers={n_layers}, n_hidden={n_hidden}, n_points_per_link={n_points_per_link}")
        else:
            n_layers, n_hidden, n_points_per_link = 20, 128, 2
            
        model = G_NDF_2D(n_layers=n_layers, n_hidden=n_hidden, n_points_per_link=n_points_per_link, device=device)
        
        # Load state dict (check if it's nested)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        print("Model loaded successfully")
        
        # Generate configuration space grid for NDF evaluation
        size = [300, 300]
        # Evaluate on the same [0, 2pi] range as the configuration space
        yy, xx = torch.meshgrid(torch.linspace(0, 2*np.pi, size[0]), 
                                torch.linspace(0, 2*np.pi, size[1]), indexing='ij')
        grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2)).float().to(device)
        
        # Evaluate NDF model - wrap to [-pi, pi] if model expectations require it
        grid_points_wrapped = (grid_points + np.pi) % (2 * np.pi) - np.pi
        with torch.no_grad():
            sdf_output = model(grid_points_wrapped)
            if sdf_output.dim() > 1:
                sdf_values = sdf_output[:, 0].reshape(size)
            else:
                sdf_values = sdf_output.reshape(size)
        
        print(f"SDF values shape: {sdf_values.shape}, range: [{sdf_values.min():.3f}, {sdf_values.max():.3f}]")
        
        # Normalize for visualization - use actual min/max to ensure both blue and red are distinct
        vmin_val = sdf_values.min().item()
        vmax_val = sdf_values.max().item()
        # Ensure vmin is negative and vmax is positive for TwoSlopeNorm to work correctly
        vmin_val = min(vmin_val, -1e-3)
        vmax_val = max(vmax_val, 1e-3)
        norm = TwoSlopeNorm(vmin=vmin_val, vcenter=0.0, vmax=vmax_val)
        
        # Plot NDF SDF heatmap
        extent = [0, 2*np.pi, 0, 2*np.pi]
        a1[2].imshow(sdf_values.cpu().numpy(), origin='lower', cmap='seismic', 
                    extent=extent, norm=norm, interpolation='nearest', alpha=0.95)
        
        a1[2].set_xlabel(r'$\theta_1$')
        a1[2].set_ylabel(r'$\theta_2$')
        a1[2].set_xlim(0, 2*np.pi)
        a1[2].set_ylim(0, 2*np.pi)
        a1[2].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
        a1[2].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
        a1[2].set_xticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        a1[2].set_yticklabels([r'$0$', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'])
        
        # Apply same styling to NDF subplot
        for label in a1[2].get_xticklabels() + a1[2].get_yticklabels():
            label.set_color('#2C3E50')
            label.set_alpha(0.8)
        a1[2].set_aspect('equal')
        a1[2].grid(which='major', axis='both', linestyle='--', alpha=0.3)
        a1[2].set_axisbelow(True)
        a1[2].minorticks_off()
        for spine in a1[2].spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('#34495E')
            spine.set_alpha(0.8)
        a1[2].xaxis.label.set_color('#2C3E50')
        a1[2].xaxis.label.set_alpha(0.9)
        a1[2].yaxis.label.set_color('#2C3E50')
        a1[2].yaxis.label.set_alpha(0.9)
        a1[2].tick_params(colors='gray', length=0, width=0)
        
        # Colorbar removed as requested
        
    except Exception as e:
        print(f"Could not load NDF model: {e}")
    
    
    # tt = np.linspace(0, 2*np.pi, 101)
    # a1[0].plot(arm_problem.robot._base_position.x + arm_problem.robot._link_lengths[0]*np.cos(tt),
    #            arm_problem.robot._base_position.y + arm_problem.robot._link_lengths[0]*np.sin(tt),
    #            '--',
    #            color='grey',
    #            lw=0.5)

    if not args.no_animation:
        # Animation
        animation_length = 10.0
        arm_anim = arm_anim = ArmAnimator(arm_problem.robot,
                                          arm_problem.workspace.obstacles,
                                          c_space,
                                          path_full,
                                          map_lims[0],
                                          map_lims[1],
                                          theta[0][[0, -1]],
                                          theta[1][[0, -1]],
                                          shadow_skip=args.arm_shadows
                                          )
                                        
        delta_t = (animation_length * 1000.0 / arm_anim.max_frames)
        arm_animation = animation.FuncAnimation(arm_anim.fig, arm_anim.animate, init_func=arm_anim.init_fig, frames=arm_anim.max_frames,
                                  interval=delta_t, blit=True)
                                
        if args.save_animation is not None:

            arm_animation.save(args.save_animation, writer='ffmpeg', fps=int(1000.0/delta_t), dpi=200,
                               extra_args=["-crf", "18", "-profile:v", "main", "-tune", "animation", "-pix_fmt", "yuv420p"])
    
    # Save the static figure (IROS publication quality)
    f1.savefig('config_space_visualization.png', dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
    f1.savefig('config_space_visualization.pdf', bbox_inches='tight', facecolor='white', edgecolor='none')
    print("Static figures saved as 'config_space_visualization.png' and 'config_space_visualization.pdf'")
    
    plt.show()