import gym
import sys
import numpy as np
import time
np.set_printoptions(threshold=sys.maxsize)

class GymEnv:
    def __init__(self, render_mode: str = None):
        if render_mode!=None:

            self.env = gym.make("Breakout-v4", render_mode=render_mode)
        else:
            self.env = gym.make("Breakout-v4")

        self.env.reset()

    def test(self) -> None:
        """
        Test the availability of the environment.
        :return:
        """
        for i in range(1000):
            action = self.env.action_space.sample()
            obs, reward, done, info = self.env.step(action)
            print("action is ", action, "episode is: ", i, done,info['lives'])
            if done:
                break

    def image_info(self)->None:
        pass



if __name__ == '__main__':
    e = GymEnv()

    print(e.test())
