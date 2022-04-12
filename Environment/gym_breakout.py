import gym
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


class GymEnv:
    def __init__(self, render_mode: str = "human"):
        self.env = gym.make('Breakout-v0', render_mode=render_mode)
        self.env.reset()

    def test(self):
        for i in range(1000):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            print("action is ", action, "episode is: ", i, done)
            if done:
                break


if __name__ == '__main__':
    e = GymEnv()
    e.test()