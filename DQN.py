import sys
import gym
import torch
import numpy as np
from collections import deque
from datetime import datetime
from copy import deepcopy
from skimage.transform import resize
from skimage.color import rgb2gray
from Model.DQN_Agent import DQNAgent
from Train.read_write_operations import ProjectIO
import time
import matplotlib.pyplot as plt

torch.cuda.empty_cache()
def find_max_lifes(env):
    env.reset()
    _, _, _, info = env.step(0)
    return info['lives']

def check_live(life, cur_life):
    if life > cur_life:
        return True
    else:
        return False

def pre_proc(X):
    x = np.uint8(resize(rgb2gray(X), (HEIGHT, WIDTH), mode='reflect') * 255)
    return x

def get_init_state(history, s):
    for i in range(HISTORY_SIZE):
        history[i, :, :] = pre_proc(s)

if __name__ == "__main__":
    EPISODES = 500000
    HEIGHT = 84
    WIDTH = 84
    HISTORY_SIZE = 4
    #
    render_mode='humam'
    render_mode=None
    if render_mode != None:
        env = gym.make('Breakout-v4', render_mode=render_mode)
    else:
        env = gym.make('Breakout-v4')
    # get the game max lifes
    max_life = find_max_lifes(env)
    state_size = env.observation_space.shape

    # action_size = env.action_space.n
    action_size = 3
    scores, episodes = [], []
    io_obj=ProjectIO()
    io_obj.write_log("start")
    config_name="train_lightning"
    agent = DQNAgent(action_size=action_size,
                     io_obj=io_obj,
                     config_name=config_name,
                     strategy="q")
    recent_reward = deque(maxlen=100)
    frame = 0
    memory_size = 0
    for e in range(EPISODES):

        done = False
        score = 0
        # initialize the input history
        history = np.zeros([5, HEIGHT, WIDTH], dtype=np.uint8)
        step = 0
        d = False
        state = env.reset()
        # drop the point board
        state = state[21:]
        life = max_life

        # compress the original frame
        get_init_state(history, state)
        # show the state
        # im=Image.fromarray(history[0])
        # im.show()

        time_start = time.time()
        while not done:

            step += 1
            frame += 1
            if agent.render:
                env.render()

            # get action for the current state and go one step in environment
            action = agent.get_action(np.float64(history[:4, :, :]) / 255.)

            next_state, reward, done, info = env.step(action + 1)

            #
            pre_proc_next_state = pre_proc(next_state)
            history[4, :, :] = pre_proc_next_state
            ter = check_live(life, info['lives'])

            life = info['lives']
            r = np.clip(reward, -1, 1)

            # save the sample <s, a, r, s'> to the replay memory
            agent.append_sample(deepcopy(pre_proc_next_state), action, r, ter)
            # every time step do the training and train until have enough samples
            if frame >= agent.train_start:
                print("start one train at frame ",frame)
                agent.train_model(frame)
                # print("check")
                if frame % agent.update_target == 0:
                    agent.update_target_model()
            score += reward
            # shift one step at tail
            history[:4, :, :] = history[1:, :, :]
            time_end=time.time()
            if frame % 500 == 0:

                # print('now time : ', datetime.now())
                scores.append(score)
                episodes.append(int(e))
                # print(episodes,scores)
                plt.plot(episodes,scores)
                plt.savefig("Results/breakout_dqn_q.png")
                plt.clf()

            if done and frame>=agent.train_start:
                recent_reward.append(score)
                # every episode, plot the play time
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon, "   steps:", step,
                      "    recent reward:", np.mean(recent_reward),
                      "    time consume:",time_end-time_start)
                # write into log fiel

                log=f"episode:{e}, score:{score}, memory length:{len(agent.memory)}, epsilon:{agent.epsilon}, " \
                    f"steps:{step}, recent reward:{np.mean(recent_reward)}, time consume:{time_end-time_start}"
                io_obj.write_log(log)

                # if the mean of scores of last 10 episode is bigger than 400
                # stop training
                if np.mean(recent_reward) > 50:
                    torch.save(agent.model, "./SavedModels/breakout_dqn")
                    sys.exit()