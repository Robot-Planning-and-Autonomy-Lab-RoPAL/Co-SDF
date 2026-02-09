import numpy as np
import polygon_tools as poly
import csv
import yaml
import sys
import math
import matplotlib as plt
from matplotlib.patches import Polygon as PlotPolygon
from matplotlib.collections import PatchCollection
import matplotlib.cm as cm
import logging


def robot_builder(robot):
    # Note that the robot type must be implemented in this module, so the example robot:
    #  {type: RobotArm2D, parameters: {base_position: [5.0, 5.0], link_lengths: [2.1, 2.1]}
    # would call as constructor: robot_tools.RobotArm2D(base_position=[5.0, 5.0], link_lengths=[2.1, 2.1])
    return getattr(sys.modules[__name__], robot['type'])(**robot['parameters'])


def workspace_builder(workspace):
    # Note that the workspace type must be implemented in this module, so the example workspace:
    #  {type: Workspace2D, parameters: {limits=[[0,1.0],[0,1.0]], obstacles=[]}
    # would call as constructor: robot_tools.Workspace2D(limits=[[0,1.0],[0,1.0]], obstacles=[])
    return getattr(sys.modules[__name__], workspace['type'])(**workspace['parameters'])


class Workspace2D(object):
    def __init__(self, limits=[[0,1.0],[0,1.0]], obstacles=[]):
        self.limits = np.array(limits)
        assert self.limits.shape == (2,2), 'Currently only implemented for 2D workspaces'
        self.obstacles = []
        for ob in obstacles:
            # Add each obstacle (must be a Polygon or derived class like Rectangle from poly_tools)
            self.obstacles.append(getattr(poly, ob['type'])(**ob['parameters']))

    def in_collision_point(self, point):
        p = poly.Point(*point)
        collision = False
        for o in self.obstacles:
            if o.point_inside(p):
                collision = True
                break
        return collision

    def in_collision_poly(self, polygon):
        collision = False
        for o in self.obstacles:
            if polygon.intersect(o):
                collision = True
                break
        return collision

    def get_clearance(self, obj):
        """Returns the signed distance between a Polygon/Point and any obstacle.
        Positive: outside, Negative: inside.
        """
        min_dist = 1e6
        for o in self.obstacles:
            if hasattr(obj, "to_polygon") and not isinstance(obj, poly.Polygon):
                obj_poly = obj.to_polygon()
            elif isinstance(obj, poly.Polygon):
                obj_poly = obj
            else:
                obj_poly = None

            if obj_poly is not None:
                # Polygon-to-obstacle distance
                dist = obj_poly.distance(o)
                # For signed distance, we'd need to check if polygon is inside o
                # Simple approximation: if any vertex is inside, it's negative
                if dist == 0:
                    # Find deepest penetration
                    max_pen = 0
                    for p in obj_poly:
                        if isinstance(o, poly.Circle):
                            pen = o.radius - math.sqrt((p.x - o.center.x)**2 + (p.y - o.center.y)**2)
                        elif isinstance(o, poly.Polygon):
                            if o.point_inside(p):
                                pen = math.sqrt(min(poly.p_to_seg_dist_sq(p, v0, v1) for v0, v1 in o.edges()))
                            else: pen = 0
                        max_pen = max(max_pen, pen)
                    dist = -max_pen
            elif isinstance(obj, poly.Point):
                # Point-to-obstacle signed distance
                if isinstance(o, poly.Circle):
                    dist = math.sqrt((obj.x - o.center.x)**2 + (obj.y - o.center.y)**2) - o.radius
                elif isinstance(o, poly.Polygon):
                    d2 = min(poly.p_to_seg_dist_sq(obj, v0, v1) for v0, v1 in o.edges())
                    dist = math.sqrt(d2)
                    if o.point_inside(obj):
                        dist = -dist
                else:
                    dist = 1e6
            else:
                dist = 1e6
                
            if dist < min_dist:
                min_dist = dist
        return min_dist

    def plot(self, hax=None, cmap=cm.viridis):
        from matplotlib.patches import Circle as PlotCircle
        if hax is None:
            f, hax = plt.subplots(1)
        
        # Plot obstacles
        for o in self.obstacles:
            if isinstance(o, poly.Polygon):
                pts = np.array([[p.x, p.y] for p in o])
                hax.add_patch(PlotPolygon(pts, color='gray', alpha=0.5, zorder=1))
            elif isinstance(o, poly.Circle):
                hax.add_patch(PlotCircle((o.center.x, o.center.y), o.radius, color='gray', alpha=0.5, zorder=1))
        
        # Set limits
        if hasattr(self, 'limits'):
            hax.set_xlim(self.limits[0])
            hax.set_ylim(self.limits[1])
        else:
            hax.set_xlim(0, 10)
            hax.set_ylim(0, 10)
        hax.set_aspect('equal')

        hax.set_xlabel(r'$x$')
        hax.set_ylabel(r'$y$')


class Robot2D(object):
    _c_poly = None

    def __init__(self, pos=[0.0, 0.0], heading=0.0, footprint=[(0.0, 0.0)], footprint_file=None):
        self.R = np.eye(2)

        if footprint_file is not None:
            with open(footprint_file, mode='r') as fh:
                csv_reader = csv.reader(fh)
                footprint = []
                for row in csv_reader:
                    assert len(row) == 2, 'Row {0} does not have 2 elements'.format(len(row)+1)
                    footprint.append([float(row[0]), float(row[1])])
            logging.info('Loaded robot footprint file {0} with {1} points'.format(footprint_file, len(footprint)))

        self.position = poly.Point(pos[0], pos[1])
        self.footprint = poly.PointList(footprint)
        self.heading = heading
        self._set_heading_transformation()
        self._update_poly()


    def _set_heading_transformation(self):
        ct, st = np.cos(self.heading), np.sin(self.heading)
        self.R = np.array([[ct, -st], [st, ct]])

    def set_heading(self, heading):
        self.heading = heading
        self._set_heading_transformation()
        self._update_poly()

    def set_position(self, pos):
        self.position = pos
        self._update_poly()

    def set_footprint(self, footprint):
        self.footprint = footprint
        self._update_poly()

    def _update_poly(self):
        self._c_poly = poly.Polygon([poly.Point(*(np.matmul(self.R, p)+self.position)) for p in self.footprint])

    def get_current_polygon(self):
        return self._c_poly


class RobotArm2D(object):
    _spine_pts = None

    def __init__(self, base_position=[0.0, 0.0], link_lengths=[1.0, 1.0], link_angles=[0.0, 0.0], link_width=0.2, joint_limits=None):
        # Assume arm angles are relative (can be summed)

        self._base_position = poly.Point(base_position[0], base_position[1])

        assert len(link_lengths) == len(link_angles)
        self._link_lengths = np.array(link_lengths)
        self._link_angles = np.array(link_angles)
        self._link_width = link_width
        
        if joint_limits is None:
            # Default to [0, 2pi] for each joint to align with current "boxed" logic
            self.joint_limits = np.array([[0.0, 2 * np.pi] for _ in range(len(link_lengths))])
        else:
            self.joint_limits = np.array(joint_limits)

        self._R = [np.eye(2) for i in self._link_angles]
        self._set_rotation_transforms()

    def set_link_angles(self, link_angles):
        self._link_angles = np.array(link_angles)
        self._set_rotation_transforms()

    def _set_rotation_transforms(self):
        sum_angles = self._link_angles.cumsum()
        for i, theta in enumerate(sum_angles):
            ct, st = np.cos(theta), np.sin(theta)
            self._R[i] = np.array([[ct, -st], [st, ct]])
        self._set_spine_points()

    def get_current_polygon(self):
        # Generate a thick polygon by offsetting the spine
        # Each link contributes its own start and end offset points
        if self._link_width <= 0:
            return poly.Polygon(self._spine_pts + self._spine_pts[-2:0:-1])
            
        pts_left = []
        pts_right = []
        
        sum_angles = self._link_angles.cumsum()
        for i, (p_start, p_end) in enumerate(zip(self._spine_pts[:-1], self._spine_pts[1:])):
            angle = sum_angles[i]
            nx = -np.sin(angle)
            ny = np.cos(angle)
            r = self._link_width / 2.0
            
            # Add both start and end offsets for THIS link
            pts_left.append(poly.Point(p_start.x + nx * r, p_start.y + ny * r))
            pts_left.append(poly.Point(p_end.x + nx * r, p_end.y + ny * r))
            
            pts_right.append(poly.Point(p_start.x - nx * r, p_start.y - ny * r))
            pts_right.append(poly.Point(p_end.x - nx * r, p_end.y - ny * r))
            
        # Order: left points (base to tip), then right points (tip to base)
        return poly.Polygon(pts_left + pts_right[::-1])

    def _set_spine_points(self):
        self._spine_pts = [self._base_position]
        for R, ll in zip(self._R, self._link_lengths):
            self._spine_pts.append(poly.Point(*(np.matmul(R, [ll, 0])+self._spine_pts[-1])))

    def get_spine_points(self):
        return [p.x for p in self._spine_pts], [p.y for p in self._spine_pts]

    def get_collision_shapes(self):
        """Returns a list of shapes (Polygons and Circles) for collision detection."""
        if self._link_width <= 0:
            return [self.get_current_polygon()]
            
        shapes = []
        r = self._link_width / 2.0
        
        # 1. Add joints as circles (including base and end-effector)
        for pt in self._spine_pts:
            shapes.append(poly.Circle(pt, r))
            
        # 2. Add links as rectangles
        sum_angles = self._link_angles.cumsum()
        for i, (p_start, p_end) in enumerate(zip(self._spine_pts[:-1], self._spine_pts[1:])):
            angle = sum_angles[i]
            nx = -np.sin(angle)
            ny = np.cos(angle)
            
            # Corner points for the rectangle representing this link
            p1 = poly.Point(p_start.x + nx * r, p_start.y + ny * r)
            p2 = poly.Point(p_end.x + nx * r, p_end.y + ny * r)
            p3 = poly.Point(p_end.x - nx * r, p_end.y - ny * r)
            p4 = poly.Point(p_start.x - nx * r, p_start.y - ny * r)
            
            shapes.append(poly.Polygon([p1, p2, p3, p4]))
            
        return shapes

    def get_end_effector_position(self):
        return self._spine_pts[-1]

    def end_effector_path(self, config_path):
        c_pose = self._link_angles.copy()
        ee_path = []
        for pose in config_path:
            self.set_link_angles(pose)
            ee_path.append(self.get_end_effector_position())
        self.set_link_angles(c_pose)
        return np.array(ee_path)


class PlanningProblem(object):

    def __init__(self, world_file):

        # Load world
        with open(world_file, 'r') as fh:
            world = yaml.safe_load(fh)

        self.workspace = workspace_builder(world['workspace'])
        # Note that the robot type must be implemented in the robot_tools module, so the example robot:
        #  {type: RobotArm2D, parameters: {base_position: [5.0, 5.0], link_lengths: [2.1, 2.1]}
        # would call as constructor: robot_tools.RobotArm2D(base_position=[5.0, 5.0], link_lengths=[2.1, 2.1])
        self.robot = robot_builder(world['robot'])


    def construct_config_space(self, nx=101):
        # TODO: This should be more general (number of dimensions, wraparound etc. in the robot class)
        t1_lim, t2_lim = self.robot.joint_limits[0], self.robot.joint_limits[1]
        theta1 = np.linspace(t1_lim[0], t1_lim[1], nx)
        theta2 = np.linspace(t2_lim[0], t2_lim[1], nx)
        v = np.zeros((len(theta1), len(theta2)), dtype=int)

        for i, t1 in enumerate(theta1):
            for j, t2 in enumerate(theta2):
                self.robot.set_link_angles([t1, t2])
                in_obs = 0
                
                # Check collision for all parts of the robot
                collision_shapes = self.robot.get_collision_shapes()
                
                for o_num, o in enumerate(self.workspace.obstacles):
                    collides = False
                    for shape in collision_shapes:
                        if shape.intersect(o):
                            collides = True
                            break
                    if collides:
                        in_obs = o_num + 1
                        break
                v[i, j] = in_obs

        return [theta1, theta2], v