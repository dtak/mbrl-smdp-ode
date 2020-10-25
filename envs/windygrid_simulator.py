import numpy as np
import gym
from gym import spaces
from gym.utils import seeding
import matplotlib.pyplot as plt

SQRT2 = np.sqrt(2)


class WindyGridSimulator(gym.Env):

    num_states = 2
    num_actions = 8
    t = 0
    horizon = 150

    def __init__(self):
        self.height = 7
        self.width = 10
        self.one_move = .17

        low = np.array([-self.height/2, -self.width/2])
        high = -low
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(8)

        self.moves = {
            0: np.array((-1, 0)) * self.one_move,  # up
            1: np.array((0, 1)) * self.one_move,   # right
            2: np.array((1, 0)) * self.one_move,   # down
            3: np.array((0, -1)) * self.one_move,  # left
            4: np.array((1, 1)) * self.one_move / SQRT2,  # upright
            5: np.array((-1, 1)) * self.one_move / SQRT2,  # upleft
            6: np.array((1, -1)) * self.one_move / SQRT2,  # downright
            7: np.array((-1, -1)) * self.one_move / SQRT2,  # downleft
        }

        self.seed()
        self.state = None

    def __repr__(self):
        return "WindyGrid_Simulator"

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, a, dt=1):
        x, y = self.state
        for _ in range(dt):
            dx, dy = self.moves[a] + self.np_random.normal(0, 0.05, 2)
            if (-2.5 <= y <= 0.5) or (2.5 <= y <= 3.5):
                dx += 0.25*self.one_move
            elif 0.5 < y < 2.5:
                dx += 0.5*self.one_move
            x, y = self.bound_state(np.array([x + dx, y + dy]))
        self.state = np.array([x, y])
        self.t += dt
        done = self.is_terminal() or self.t >= self.horizon
        reward = self.calc_reward(dt=dt)
        return self.state, reward, done, {}

    def reset(self):
        self.t = 0
        self.state = np.array([self.np_random.uniform(-0.5, 0.5), self.np_random.uniform(-5, -4)])
        return self.state

    def is_terminal(self, state=None):
        if state is None:
            state = self.state
        else:
            state = self.bound_state(state)
        x, y = state
        return bool((-0.5 <= x <= 0.5) and (1.5 <= y <= 2.5))

    def calc_reward(self, action=0, state=None, dt=1):
        reward = -dt
        if self.is_terminal(state=state):
            reward += 10
        return reward

    def get_time_gap(self, action=0, state=None):
        return self.np_random.randint(1, 8)

    def get_time_info(self):
        return 1, 7, self.horizon, False  # min_t, max_t, max time length, is continuous

    def bound_state(self, state):
        return np.array([np.clip(state[0], -self.height/2, self.height/2),
                         np.clip(state[1], -self.width/2, self.width/2)])

    def draw(self, wind=False, action=False):
        plt.figure(figsize=(7, 7))
        plt.xticks(range(self.width + 1))
        plt.yticks(range(self.height + 1))
        plt.ylabel('$x$', fontsize=15)
        plt.xlabel('$y$', fontsize=15)
        plt.text(0.3, 3.45, '$S$', va='center', fontsize=25)
        plt.text(6.75, 3.45, '$G$', va='center', fontsize=25)
        plt.grid(True, alpha=0.5)
        plt.xlim(0, 10)
        plt.ylim(0, 7)
        plt.gca().set_aspect('equal', adjustable='box')
        if wind:
            plt.axvspan(2.5, 5.5, alpha=0.1, color='b')
            plt.axvspan(7.5, 8.5, alpha=0.1, color='b')
            plt.axvspan(5.5, 7.5, alpha=0.2, color='b')
            plt.arrow(4, 2.2, 0, 2, width=0.2, head_width=0.4, head_length=0.3, alpha=0.4, color='k')
            plt.arrow(8.3, 2.2, 0, 2, width=0.2, head_width=0.4, head_length=0.3, alpha=0.4, color='k')
            plt.arrow(6.4, 2.2, 0, 2, width=0.2, head_width=0.4, head_length=0.3, alpha=0.8, color='k')
        if action:
            for dir in self.moves.values():
                plt.arrow(1.3, 5.5, *(dir/self.one_move*0.6), width=0.01, head_width=0.1, head_length=0.1, color='k')
                plt.text(0.8, 6.5, 'Action', va='center', fontsize=13, family='sans-serif')
        plt.show()


if __name__ == '__main__':
    env = WindyGridSimulator()
    counts, rs, dts = [0]*1000, [0]*1000, []
    xs, ys = [], []
    dones = 0
    for i in range(1000):
        done = False
        s = env.reset()
        while not done:
            a = env.action_space.sample()
            dt = env.get_time_gap()
            s, r, done, info = env.step(a, dt)
            counts[i] += 1
            rs[i] += r
            dts.append(dt)
            xs.append(s[0])
            ys.append(s[1])
            if env.is_terminal(s):
                dones += 1
    print(dones)
    plt.hist(counts)
    plt.show()
    plt.hist(rs)
    plt.show()
    plt.hist(dts)
    plt.show()
    plt.scatter(ys, xs, alpha=0.05)
    plt.show()
