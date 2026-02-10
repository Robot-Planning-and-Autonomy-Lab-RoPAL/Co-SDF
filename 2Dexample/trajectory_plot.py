#!/usr/bin/env python3
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
from matplotlib.ticker import MultipleLocator
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
    raise TypeError("Unsupported obstacle type for plotting; need Polygon, Circle(with to_polygon), or Nx2 array.")


def plot_config_space(ax,
                      obstacles,
                      arm,
                      cspace_array,
                      col_map,
                      xlim,
                      ylim,
                      theta1_lim,
                      theta2_lim):
    h_obs = []
    for o in obstacles:
        xy = shape_to_xy(o, n=256)
        h_obs.append(PlotPolygon(xy, closed=True, zorder=1))

    c_obs = PatchCollection(h_obs, cmap=col_map)
    c_obs.set_array(np.linspace(0, 1.0, len(obstacles) + 1)[1:])
    ax[0].add_collection(c_obs)

    h_arm, = ax[0].plot(*arm.get_spine_points(), c='black', lw=3.0)

    for a in ax:
        a.set_aspect('equal')

    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_xlim(xlim)
    ax[0].set_ylim(ylim)

    ax[1].set_xlabel('θ1')
    ax[1].set_ylabel('θ2')
    ax[1].set_xlim(theta1_lim[0], theta1_lim[-1])
    ax[1].set_ylim(theta2_lim[0], theta2_lim[-1])

    ax[0].xaxis.set_major_locator(MultipleLocator(2.0))
    ax[0].xaxis.set_minor_locator(MultipleLocator(1.0))
    ax[0].yaxis.set_major_locator(MultipleLocator(2.0))
    ax[0].yaxis.set_minor_locator(MultipleLocator(1.0))
    ax[1].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi])
    ax[1].set_yticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax[1].xaxis.set_minor_locator(MultipleLocator(np.pi/4))
    ax[1].yaxis.set_minor_locator(MultipleLocator(np.pi/4))

    ax[1].set_xticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    ax[1].set_yticklabels(['0', 'π/2', 'π', '3π/2', '2π'])
    for a in ax:
        a.grid(which='both', axis='both')
        a.set_axisbelow(True)

    cspace_masked = np.ma.masked_where(cspace_array == 0.0, cspace_array)
    col_map.set_bad(color='white')
    ax[1].imshow(cspace_masked.T, origin='lower', cmap=col_map,
                 extent=[theta1_lim[0], theta1_lim[-1], theta2_lim[0], theta2_lim[-1]])

    return h_arm


class ArmAnimator:
    def __init__(self, arm, obstacles, cspace_array, path, x_lim, y_lim, t1_lim, t2_lim, col_map=None, shadow_skip=0):
        self.fig, self.ax = plt.subplots(1, 2)
        self.fig.set_size_inches([9.6, 5.4])
        self.arm = arm
        self.obstacles = obstacles
        self.cspace_array = cspace_array
        self.path = path
        self.cmap = (col_map or cm.viridis.copy())
        self.x_lim = x_lim
        self.y_lim = y_lim
        self.t1lim = t1_lim
        self.t2lim = t2_lim
        self.max_frames = self.path.shape[0]
        self.last_break = 0
        self._shadow_skip = shadow_skip
        self.h_arm = None
        self.plot_artists = []

    def init_fig(self):
        for a in self.ax:
            a.cla()
        self.arm.set_link_angles(self.path[0])
        self.h_arm = plot_config_space(self.ax, self.obstacles, self.arm, self.cspace_array, self.cmap,
                                       self.x_lim, self.y_lim, self.t1lim, self.t2lim)
        self.h_path, = self.ax[1].plot(self.path[:1, 0], self.path[:1, 1], 'r--')
        self.h_pathend, = self.ax[1].plot(self.path[0, 0], self.path[0, 1], 'ro')
        self.plot_artists = [self.h_arm, self.h_path, self.h_pathend]
        return self.plot_artists

    def animate(self, i):
        if self._shadow_skip != 0 and i % self._shadow_skip == 0:
            gv = 0.9 - float(i) / self.max_frames * 0.9
            h_arm_shadow = self.ax[0].plot(*self.arm.get_spine_points(), c=[gv, gv, gv], lw=1.0)
            h_arm_shadow.extend(self.plot_artists)
            self.plot_artists = h_arm_shadow

        self.arm.set_link_angles(self.path[i])
        self.h_arm.set_data(*self.arm.get_spine_points())

        if i > 0 and any(abs(self.path[i] - self.path[i-1]) > np.pi):
            old_path, = self.ax[1].plot(self.path[self.last_break:i, 0], self.path[self.last_break:i, 1], 'r--')
            self.plot_artists.append(old_path)
            self.last_break = i

        self.h_path.set_data(self.path[self.last_break:(i+1), 0], self.path[self.last_break:(i+1), 1])
        self.h_pathend.set_data([self.path[i, 0]], [self.path[i, 1]])
        return self.plot_artists


def render_gif(path_file, world_file, gif_file, nx, fps, shadows):
    print(f"Loading path from {path_file}...")
    path = np.load(path_file)
    print(f"Loading world from {world_file}...")
    arm_problem = robot_tools.PlanningProblem(world_file)
    theta, c_space = arm_problem.construct_config_space(nx)
    t1_min, t1_max = float(theta[0][0]), float(theta[0][-1])
    t2_min, t2_max = float(theta[1][0]), float(theta[1][-1])

    try:
        map_lims = arm_problem.workspace.limits
    except KeyError:
        map_lims = [[0, 10], [0, 10]]

    arm_problem.robot.set_link_angles(path[0])
    anim = ArmAnimator(arm_problem.robot,
                       arm_problem.workspace.obstacles,
                       c_space,
                       path,
                       map_lims[0],
                       map_lims[1],
                       [t1_min, t1_max],
                       [t2_min, t2_max],
                       shadow_skip=shadows)

    delta_t = (1000.0 / fps)
    ani = animation.FuncAnimation(anim.fig, 
                                  anim.animate, 
                                  init_func=anim.init_fig, 
                                  frames=anim.max_frames,
                                  interval=delta_t, 
                                  blit=True)

    from matplotlib.animation import PillowWriter
    fps = args.fps
    print(f"Saving GIF to {gif_file} (fps={fps})...")
    ani.save(gif_file, writer=PillowWriter(fps=fps))
    plt.close(anim.fig)
    print("GIF saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="2D path GIF renderer (arm-style)")
    parser.add_argument("--path", type=str, default="sdf", help="Path")
    parser.add_argument("--world", type=str, default="obs20", help="YAML world file")
    parser.add_argument("--gif-nx", type=int, default=201, help="Config-space resolution for GIF background")
    parser.add_argument("--fps", type=int, default=15, help="GIF frames per second")
    parser.add_argument("--gif-shadows", type=int, default=0, help="Arm shadows every n steps (0 off)")
    args = parser.parse_args()

    rrt_dir = "output/traj/" + args.world + "/rrt.npy"
    ndf_dir = "output/traj/" + args.world + "/ndf.npy"
    sdf_dir = "output/traj/" + args.world + "/sdf.npy"

    rrt_out = "output/traj/" + args.world + "/rrt.gif"
    ndf_out = "output/traj/" + args.world + "/ndf.gif"
    sdf_out = "output/traj/" + args.world + "/sdf.gif"

    sence = "config/" + args.world + ".yaml" 

    render_gif(rrt_dir, sence, rrt_out, nx=args.gif_nx, fps=args.fps, shadows=args.gif_shadows)
    render_gif(ndf_dir, sence, ndf_out, nx=args.gif_nx, fps=args.fps, shadows=args.gif_shadows)
    render_gif(sdf_dir, sence, sdf_out, nx=args.gif_nx, fps=args.fps, shadows=args.gif_shadows)