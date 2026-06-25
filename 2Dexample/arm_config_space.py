import os
import torch
import shutil
import argparse
import robot_tools
import numpy as np
import matplotlib.pyplot as plt
import polygon_tools as poly
import matplotlib.animation as animation
import matplotlib.cm as cm
import matplotlib.colors
import matplotlib.patheffects as path_effects
from matplotlib.colors import TwoSlopeNorm
from common.models import load_model
from cspace_sdf import cobs_sdf_on_torus
from matplotlib.patches import Polygon as PlotPolygon, Circle as PlotCircle
from matplotlib.collections import PatchCollection
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
    'font.size': 30,
    'axes.labelsize': 30,
    'axes.titlesize': 30,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'legend.fontsize': 30,
    'lines.linewidth': 2.0,
    'axes.linewidth': 1.0,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3,
})

colour_map = matplotlib.colors.ListedColormap([[0.20, 0.50, 0.70, 1.0]])

AXIS_TEXT_COLOR = '#4A4A4A'
AXIS_SPINE_COLOR = '#7A7A7A'
WORKSPACE_BG = '#F2F1EC'
CSPACE_BG = '#F0EFE9'


def _set_theta_ticks(ax, single_origin_zero=True):
    if single_origin_zero:
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_yticks([np.pi, 2 * np.pi])
        ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
        ax.set_yticklabels([r'$\pi$', r'$2\pi$'])
    else:
        ax.set_xticks([0, np.pi, 2 * np.pi])
        ax.set_yticks([0, np.pi, 2 * np.pi])
        ax.set_xticklabels([r'$0$', r'$\pi$', r'$2\pi$'])
        ax.set_yticklabels([r'$0$', r'$\pi$', r'$2\pi$'])


def _style_axes(ax, *, grid=False):
    ax.set_axisbelow(True)
    ax.minorticks_off()
    ax.grid(grid, which='major', axis='both', linestyle='--', alpha=0.18 if grid else 0.0)

    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.9)
        ax.spines[side].set_color(AXIS_SPINE_COLOR)
        ax.spines[side].set_alpha(0.7)

    ax.tick_params(colors=AXIS_TEXT_COLOR, length=3.0, width=0.8, pad=3)
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_color(AXIS_TEXT_COLOR)
        label.set_alpha(0.9)


def _style_workspace_axes(ax):
    ax.set_facecolor(WORKSPACE_BG)
    _style_axes(ax, grid=False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.tick_params(length=0)


def _style_cspace_axes(ax):
    ax.set_facecolor(CSPACE_BG)
    _style_axes(ax, grid=False)


def _ticks_without_zero(lim, step=2.0, keep_zero=False):
    lo, hi = float(lim[0]), float(lim[1])
    start = np.ceil(lo / step) * step
    end = np.floor(hi / step) * step
    if end < start:
        return []
    ticks = np.arange(start, end + 0.5 * step, step)
    if keep_zero:
        return list(ticks)
    return [t for t in ticks if not np.isclose(t, 0.0)]


def _draw_checkerboard_background(ax, xlim, ylim, cell=2.0):
    x0, x1 = float(xlim[0]), float(xlim[1])
    y0, y1 = float(ylim[0]), float(ylim[1])
    nx = max(1, int(np.ceil((x1 - x0) / cell)))
    ny = max(1, int(np.ceil((y1 - y0) / cell)))
    board = (np.add.outer(np.arange(nx), np.arange(ny)) % 2).T
    checker_cmap = matplotlib.colors.ListedColormap([WORKSPACE_BG, '#ECEBE4'])
    ax.imshow(
        board,
        origin='lower',
        extent=[x0, x1, y0, y1],
        interpolation='nearest',
        cmap=checker_cmap,
        alpha=1.0,
        zorder=0,
    )

def angle_wrap(angles):
    return angles % (2 * np.pi)

def linear_path(points, n=50):
    c_point = points[0]
    path = np.array([c_point])

    for end_point in points[1:]:
        new_path = np.linspace(path[-1], end_point, n)
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
    
    # Plot obstacles in workspace
    h_obs = []
    for o in obstacles:
      xy = shape_to_xy(o, n=256)
      h_obs.append(PlotPolygon(xy, closed=True, zorder=1))
        
    c_obs = PatchCollection(h_obs, cmap=col_map, edgecolors='#2E5E86', linewidths=0.7)
    c_obs.set_array(np.linspace(0, 1.0, len(obstacles) + 1)[1:])
    c_obs.set_alpha(0.9)
    c_obs.set_path_effects([
        path_effects.withSimplePatchShadow(offset=(0.8, -0.8), shadow_rgbFace=(0, 0, 0), alpha=0.10),
        path_effects.Normal(),
    ])
    ax[0].add_collection(c_obs)

    # Plot Robot using exact geometry (Collision Shapes)
    shapes = arm.get_collision_shapes()
    spine_points = arm.get_spine_points()
    
    # Visual artists for animation/updates
    h_arm_parts = []
    h_joints = []
    h_ee = None
    
    for shape in shapes:
        if isinstance(shape, poly.Circle):
            # Check if it's the End-Effector (last spine point)
            is_ee = (abs(shape.center.x - spine_points[0][-1]) < 1e-5 and 
                     abs(shape.center.y - spine_points[1][-1]) < 1e-5)
            c = 'black' if is_ee else '#E74C3C' # Black EE, Red Joints
            patch = PlotCircle((shape.center.x, shape.center.y), shape.radius, 
                              facecolor=c, edgecolor='white', linewidth=0.7, zorder=5)
            patch.set_path_effects([
                path_effects.withSimplePatchShadow(offset=(0.45, -0.45), shadow_rgbFace=(0, 0, 0), alpha=0.10),
                path_effects.Normal(),
            ])
            ax[0].add_patch(patch)
            if is_ee: h_ee = patch
            else: h_joints.append(patch)
        else:
            # Link body
            patch = PlotPolygon(shape_to_xy(shape), facecolor='#F39C12', edgecolor='#C47E0D',
                               linewidth=0.8, alpha=0.98, zorder=4)
            patch.set_path_effects([
                path_effects.withSimplePatchShadow(offset=(0.6, -0.6), shadow_rgbFace=(0, 0, 0), alpha=0.10),
                path_effects.Normal(),
            ])
            ax[0].add_patch(patch)
            h_arm_parts.append(patch)

    for a in ax:
        a.set_aspect('equal')

    # Remove titles and labels for paper
    # ax[0].set_xlabel(r'$x$')
    # ax[0].set_ylabel(r'$y$')
    ax[1].set_xlabel(r'$\theta$1')
    ax[1].set_ylabel(r'$\theta$2')

    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)
    _draw_checkerboard_background(ax[0], xlim, ylim, cell=2.0)
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_xticks(_ticks_without_zero(xlim, step=2.0, keep_zero=True))
    ax[0].set_yticks(_ticks_without_zero(ylim, step=2.0, keep_zero=False))
    
    ax[1].set_xlim(theta1_lim[0], theta1_lim[-1])
    ax[1].set_ylim(theta2_lim[0], theta2_lim[-1])
    _set_theta_ticks(ax[1], single_origin_zero=True)
    _style_workspace_axes(ax[0])
    _style_cspace_axes(ax[1])

    cspace_binary = (np.asarray(cspace_array) != 0).astype(float)
    cspace_array = np.ma.masked_where(cspace_binary == 0.0, cspace_binary)
    col_map.set_bad(color=CSPACE_BG)
    ax[1].imshow(cspace_array.transpose(), origin='lower', cmap=col_map,
                 extent=[theta1_lim[0], theta1_lim[1], theta2_lim[0], theta2_lim[1]],
                 interpolation='nearest', alpha=0.92)
    # ax[1].contour(
    #     cspace_binary.transpose(),
    #     levels=[0.5],
    #     colors=['#17324A'],
    #     linewidths=1.0,
    #     alpha=0.95,
    #     origin='lower',
    #     extent=[theta1_lim[0], theta1_lim[1], theta2_lim[0], theta2_lim[1]],
    # )
    
    if heat is not None:
        hm = ax[1].imshow(heat.T, origin='lower', extent=heat_extent,
                          cmap=(heat_cmap or "RdBu_r"), norm=heat_norm)

        ax[1]._heatmap_handle = hm

    # Return arm parts for animation updates
    return h_arm_parts, h_joints, h_ee

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
        self.fig.set_size_inches([9.6, 5.4])    
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
        self.plot_artists = [self.h_arm] + self.h_joints + [self.h_ee, self.h_path, self.h_pathend, self.h_ee_path, self.h_ee_pathend]

        return self.plot_artists

    def animate(self, i):

        if self._shadow_skip != 0 and i % self._shadow_skip == 0:
            gv = 0.9-float(i)/self.max_frames*0.9
            h_arm_shadow = self.ax[0].plot(*self.arm.get_spine_points(), c=[gv, gv, gv], lw=1.5)
            h_arm_shadow.extend(self.plot_artists)
            self.plot_artists = h_arm_shadow

        self.arm.set_link_angles(self.path[i])
        arm_poly = self.arm.get_current_polygon()
        self.h_arm.set_xy(shape_to_xy(arm_poly))
        spine_points = self.arm.get_spine_points()
        
        for j, (px, py) in enumerate(zip(spine_points[0][:-1], spine_points[1][:-1])):
            self.h_joints[j].center = (px, py)
            
        self.h_ee.center = (spine_points[0][-1], spine_points[1][-1])

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
    parser.add_argument('--nx', type=int, default=201, help='Resolution (n points in each dimension)')
    parser.add_argument('--world', default='block_world4', help='World name')
    parser.add_argument('--save-animation', default=None, help='Save animation to file')
    parser.add_argument('--no-animation', default=False, help='Don\'t animate')
    parser.add_argument('--arm-shadows', type=int, default=0, help='Plot shadows of arm position every n steps (0 for off)')
    args = parser.parse_args()

    sence = 'config/' + args.world + '.yaml'
    arm_problem = robot_tools.PlanningProblem(sence)
    theta, c_space = arm_problem.construct_config_space(args.nx)
    t1_min, t1_max = float(theta[0][0]), float(theta[0][-1])
    t2_min, t2_max = float(theta[1][0]), float(theta[1][-1])
    
    path_fit = np.polyfit([0.3, 1.6, 4.3, 5.9], [1.2, 0.8, 3.3, 3.2], 3)
    path_theta2 = np.linspace(0.3, 5.9, 300)
    path_theta1 = np.polyval(path_fit, path_theta2)
    path_full = angle_wrap(np.array([path_theta1, path_theta2]).T)

    ee_path = arm_problem.robot.end_effector_path(path_full)

    f_ws, ax_ws = plt.subplots(figsize=(6, 6), layout='constrained')
    f_cs, ax_cs = plt.subplots(figsize=(6, 6), layout='constrained')
    f_ndf, ax_ndf = plt.subplots(figsize=(6, 6), layout='constrained')
    a1 = [ax_ws, ax_cs, ax_ndf]
    
    figs = [f_ws, f_cs, f_ndf]
    names = ['workspace.png', 'cspace.png', 'ndf_cspace.png']
    names = [os.path.join('output', 'visu', args.world, n) for n in names]

    for ax in a1:
        ax.set_aspect('equal')

    arm_problem.robot.set_link_angles(path_full[0])
    try:
        map_lims = arm_problem.workspace.limits
    except KeyError:
        map_lims = [[0, 10], [0, 10]]

    initial_pose = np.array([np.pi/4, np.pi/3])
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
    
    # a1[1].plot(initial_pose[0], initial_pose[1], 'o', color='#E74C3C', markersize=8, 
    #           markeredgewidth=1.5, markeredgecolor='white', zorder=10)
    
    # plt.setp(a1[1].get_xticklabels(), visible=False)
    # a1[1].set_xlabel('')
    
    model_path = f'output/{args.world}/model_final.pt'
    if os.path.exists(model_path):
        try:
            device = 'cpu'
            
            model = load_model(model_path, device)
            model.eval()
            
            size = [300, 300]
            t1_lim, t2_lim = arm_problem.robot.joint_limits[0], arm_problem.robot.joint_limits[1]
                
            yy, xx = torch.meshgrid(torch.linspace(t1_lim[0], t1_lim[1], size[0]), 
                                    torch.linspace(t2_lim[0], t2_lim[1], size[1]), indexing='ij')
            grid_points = torch.stack([xx, yy], axis=2).reshape((-1, 2)).float().to(device)
                
            with torch.no_grad():
                sdf_output = model(grid_points)
                if sdf_output.dim() > 1:
                    sdf_values = sdf_output[:, 0].reshape(size)
                else:
                    sdf_values = sdf_output.reshape(size)
                
                vmin_val = sdf_values.min().item()
                vmax_val = sdf_values.max().item()
                vmin_val = min(vmin_val, -1e-3)
                vmax_val = max(vmax_val, 1e-3)
                norm = TwoSlopeNorm(vmin=vmin_val, vcenter=0.0, vmax=vmax_val)
                
                extent = [t1_lim[0], t1_lim[1], t2_lim[0], t2_lim[1]]
                a1[2].imshow(sdf_values.cpu().numpy(), origin='lower', cmap='seismic', 
                            extent=extent, norm=norm, interpolation='nearest', alpha=0.95)
                
                a1[2].set_xlabel(r'$\theta$1')
                a1[2].set_ylabel(r'$\theta$2')
                a1[2].set_xlim(extent[0], extent[1])
                a1[2].set_ylim(extent[2], extent[3])
                _set_theta_ticks(a1[2], single_origin_zero=True)
                a1[2].set_aspect('equal')
                _style_cspace_axes(a1[2])
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"Could not load NDF model: {e}")
    
    if not args.no_animation:
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
    for f, name in zip(figs, names):
        os.makedirs(os.path.dirname(name), exist_ok=True)
        f.savefig(name, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
        print(f"Saved: {name}")
    
    plt.show()
