import os
import mujoco
import numpy as np


class JacoEnv():
    def __init__(self,
                 width,
                 height,
                 frame_skip=1,
                 model_path="jaco.xml"):
        self.frame_skip = frame_skip
        self.width = width
        self.height = height

        # Instantiate Mujoco model
        fullpath = os.path.join(
            os.path.dirname(__file__), "assets", model_path)
        if not os.path.exists(fullpath):
            raise IOError("File %s does not exist" % fullpath)
        self.model = mujoco.MjModel.from_xml_path(fullpath)
        self.data = mujoco.MjData(self.model)

        self.init_qpos = self.data.qpos.ravel().copy()
        self.init_qvel = self.data.qvel.ravel().copy()

        # Renderer for camera views
        self.renderer = mujoco.Renderer(self.model, self.height, self.width)

        # Setup actuators
        self.actuator_bounds = self.model.actuator_ctrlrange
        self.actuator_low = self.actuator_bounds[:, 0]
        self.actuator_high = self.actuator_bounds[:, 1]
        self.actuator_ctrlrange = self.actuator_high - self.actuator_low
        self.num_actuators = len(self.actuator_low)

        # init model_data_ctrl
        self.null_action = np.zeros(self.num_actuators)
        self.data.ctrl[:] = self.null_action

        self.seed()

        self.observation_space = ((0, ), (height, width, 3), (height, width, 3))

        self.reset()

    def seed(self, seed=None):
        self.np_random = np.random.RandomState(seed)
        return [seed]

    def set_qpos_qvel(self, qpos, qvel):
        assert qpos.shape == (self.model.nq, ) and qvel.shape == (
            self.model.nv, )
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        mujoco.mj_forward(self.model, self.data)

    def reset(self):
        #  Fixed initial position of Jaco
        qpos = self.init_qpos
        qvel = self.init_qvel

        # set initial joint positions and velocities
        self.set_qpos_qvel(qpos, qvel)

        return self.get_obs()

    def render(self, camera_name=None):
        self.renderer.update_scene(self.data, camera=camera_name)
        return self.renderer.render()

    def _get_obs_joint(self):
        return np.concatenate(
            [self.data.qpos.flat[:], self.data.qvel.flat[:]])

    def _get_obs_rgb_view1(self):
        obs_rgb_view1 = self.render(camera_name='view1')
        return obs_rgb_view1

    def _get_obs_rgb_view2(self):
        obs_rgb_view2 = self.render(camera_name='view2')
        return obs_rgb_view2

    def get_obs(self):
        return (self._get_obs_joint(), self._get_obs_rgb_view1(),
                self._get_obs_rgb_view2())

    def do_simulation(self, ctrl):
        '''Do one step of simulation, taking new control as target

        Arguments:
            ctrl {np.array(num_actuator)}  -- new control to send to actuators
        '''
        ctrl = np.min((ctrl, self.actuator_high), axis=0)
        ctrl = np.max((ctrl, self.actuator_low), axis=0)

        self.data.ctrl[:] = ctrl

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

    def step(self, ctrl):
        # Do one step of simulation
        self.do_simulation(ctrl)
        return self.get_obs(), 0, False
