import numpy as np
import torch
from gym.envs.mujoco.hopper import HopperEnv


class HopperSimulator(HopperEnv):
    num_states = 12
    num_actions = 3
    t = 0
    horizon = 1e4  # a large number such that 1000 timesteps will not exceed it
    min_t = 1
    max_t = 7
    ctrl_coef = -1e-3

    def __init__(self):
        super(HopperSimulator, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def __repr__(self):
        return "Hopper_Simulator"

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10),
            self.sim.data.qpos.flat[:1]
        ])

    def step(self, a, dt=4):
        self.frame_skip = int(dt)
        ob, reward, done, info = super().step(a)
        self.t += dt
        done = done or self.t >= self.horizon
        return ob, reward, done, {}

    def reset(self):
        self.t = 0
        return super().reset()

    def calc_reward(self, action=np.zeros(3), state=None, prev_state=None, dt=4):
        assert len(action) == 3
        if state is None:
            state = self._get_obs()
        alive_bonus = 1.0
        reward = (state[-1] - prev_state[-1]) / dt
        reward += alive_bonus
        reward += self.ctrl_coef * np.square(action).sum()
        return reward

    def is_terminal(self, state=None):
        if state is None:
            state = self._get_obs()
        height, ang = state[:2]
        return (not (np.isfinite(state).all() and (np.abs(state[1:]) < 100).all() and
                     (height > .7) and (abs(ang) < .2))) or self.t >= self.horizon

    def get_time_gap(self, action=np.zeros(3), state=None):
        assert len(action) == 3
        if state is None:
            state = self._get_obs()
        amp = (self.max_t - self.min_t) // 2
        return np.clip(round(amp * np.cos(20 * np.pi * np.linalg.norm(state[5:-1])) + amp + 1), self.min_t, self.max_t)

    def get_time_info(self):
        return self.min_t, self.max_t, self.horizon, False  # min_t, max_t, max time length, is continuous

    def calc_reward_in_batch(self, states, actions, dts):
        K, H, _ = actions.size()
        next_states = states[:, 1:, :]
        heights, angs = next_states[:, :, 0], next_states[:, :, 1]
        masks = (torch.isfinite(next_states).all(dim=-1)) \
                & (torch.abs(next_states[:, :, 1:]) < 100).all(dim=-1) \
                & (heights > .7) & (torch.abs(angs) < .2)
        masks = masks.cumprod(dim=-1).bool()
        assert masks.size() == (K, H)
        rewards = torch.zeros(K, H, dtype=torch.float, device=self.device)
        rewards_alive = torch.ones(K, H, dtype=torch.float, device=self.device)
        rewards_run = (states[:, 1:, -1] - states[:, :-1, -1]) / dts
        rewards_ctrl = self.ctrl_coef * torch.sum(actions ** 2, dim=-1)
        rewards[masks] += rewards_alive[masks] + rewards_run[masks] + rewards_ctrl[masks]
        return rewards, masks
