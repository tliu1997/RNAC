import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from typing import Optional, List, Tuple


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}


class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(
        self,
        xml_file="half_cheetah.xml",
        forward_reward_weight=1.0,
        ctrl_cost_weight=0.1,
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
    ):
        utils.EzPickle.__init__(**locals())

        self._forward_reward_weight = forward_reward_weight

        self._ctrl_cost_weight = ctrl_cost_weight

        self._reset_noise_scale = reset_noise_scale

        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )
        # save nominal values
        self.gravity = -9.81
        
        self.bthigh_joint_stiffness = 240.0
        self.bshin_joint_stiffness = 180.0
        self.bfoot_joint_stiffness = 120.0
        self.fthigh_joint_stiffness = 180.0
        self.fshin_joint_stiffness = 120.0
        self.ffoot_joint_stiffness = 60.0
        
        self.actuator_ctrlrange = (-1.0, 1.0)
        self.actuator_ctrllimited = int(1)
        
        self.bthigh_joint_damping = 6.0
        self.bshin_joint_damping = 4.5
        self.bfoot_joint_damping = 3.0
        self.fthigh_joint_damping = 4.5
        self.fshin_joint_damping = 3.0 
        self.ffoot_joint_damping = 1.5

        mujoco_env.MujocoEnv.__init__(self, xml_file, 5)

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    def step(self, action_all):
        x_position_before = self.sim.data.qpos[0]
        # add noise to action for next state -> stochastic model
        if action_all.shape[0] == 12:
            action = action_all[0:6]
            noise = action_all[6:12]
        else:
            action = action_all
            noise_low = -self._reset_noise_scale
            noise_high = self._reset_noise_scale
            noise = self.np_random.uniform(low=noise_low, high=noise_high, size=action.shape)
        action_noise = action + noise
        self.do_simulation(action_noise, self.frame_skip)
        x_position_after = self.sim.data.qpos[0]
        x_velocity = (x_position_after - x_position_before) / self.dt

        ctrl_cost = self.control_cost(action)

        forward_reward = self._forward_reward_weight * x_velocity

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        info = {
            "x_position": x_position_after,
            "x_velocity": x_velocity,
            "reward_run": forward_reward,
            "reward_ctrl": -ctrl_cost,
            "noise": noise
        }

        return observation, reward, done, info

    def _get_obs(self):
        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        if self._exclude_current_positions_from_observation:
            position = position[1:]

        observation = np.concatenate((position, velocity)).ravel()
        return observation

    def reset(self,
        x_pos: float = 0.0, 
        state: Optional[int] = None
    ):
        ob = super().reset(state, x_pos)    
        return ob

    def reset_model(self, state, x_pos):
        if state is not None:
            qpos = state[0: self.model.nq-1]
            qvel = state[self.model.nq-1: self.model.nq+self.model.nv-1]
            qpos = np.concatenate((x_pos, qpos)).ravel()
        else:
            noise_low = -self._reset_noise_scale
            noise_high = self._reset_noise_scale

            qpos = self.init_qpos + self.np_random.uniform(
                low=noise_low, high=noise_high, size=self.model.nq
            )
            qvel = self.init_qvel + self._reset_noise_scale * self.np_random.randn(
                self.model.nv
            )

        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
        
    def viewer_setup(self):
        for key, value in DEFAULT_CAMERA_CONFIG.items():
            if isinstance(value, np.ndarray):
                getattr(self.viewer.cam, key)[:] = value
            else:
                setattr(self.viewer.cam, key, value)
