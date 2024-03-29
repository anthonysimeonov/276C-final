import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from pathlib import Path

import os
#****************************************
#Customize this to match path on computer
#****************************************

xml_directory = str(os.path.dirname(os.path.realpath(__file__))) + '/assets/'#str(Path.home()) + '/Desktop/my_envs/my_envs/envs/assets/' 

#****************************************
#****************************************


class ReacherSpringyEnv2(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        utils.EzPickle.__init__(self)
        mujoco_env.MujocoEnv.__init__(self, xml_directory + 'reacher_springy2.xml', 2)

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(a).sum()
        reward = reward_dist + reward_ctrl
        self.do_simulation(a, self.frame_skip)
            # def do_simulation(self, ctrl, n_frames):
            #     self.sim.data.ctrl[:] = ctrl
            #     for _ in range(n_frames):
            #     self.sim.step()

            #     self.sim = mujoco_py.MjSim(self.model)
            #     from mujoco_py
            #     MjSim = cymj.MjSim


        ob = self._get_obs()
        done = False
        return ob, reward, done, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0

    def reset_model(self):
        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        while True:
            self.goal = self.np_random.uniform(low=-.2, high=.2, size=2)
            if np.linalg.norm(self.goal) < 2:
                break
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        theta = self.sim.data.qpos.flat[:2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat[:2],
            self.get_body_com("fingertip") - self.get_body_com("target")
        ])
