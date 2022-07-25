import sys
import gym
import torch
import numpy as np
from collections import deque
from copy import deepcopy

from PIL import Image
from skimage.transform import resize
from skimage.color import rgb2gray
from Model.DQN_Agent import DQNAgent
from Train.read_write_operations import ProjectIO
import time
import matplotlib.pyplot as plt
import sys,getopt
import pennylane as qml

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

def get_input_parameters():
    model_type = None
    config_name = None
    encode_mode = None
    argv = sys.argv[1:]

    try:
        opts, args = getopt.getopt(argv, "m:c:e:", ["model=", "config=", "encode="])
    except:
        print("Error")

    for opt, arg in opts:
        if opt in ['--model']:
            model_type = arg
        elif opt in ['--config']:
            config_name = arg
        elif opt in ['--encode']:
            encode_mode = arg

    return [model_type,config_name,encode_mode]

@qml.qnode("lightning.qubit", interface='torch',diff_method="parameter-shift")
def encode(inputs,)


if __name__ == '__main__':
    EPISODES = 500000
    HEIGHT = 84
    WIDTH = 84
    HISTORY_SIZE = 4
    env = gym.make('Breakout-v4')
    # get the game max lifes
    max_life = find_max_lifes(env)
    state_size = env.observation_space.shape

    # action_size = env.action_space.n
    action_size = 3
    scores, episodes = [], []
    recent_reward = deque(maxlen=100)
    frame = 0
    start_train_tag = False

    done = False
    score = 0
    # initialize the input history
    history = np.zeros([5, HEIGHT, WIDTH], dtype=np.uint8)
    step = 0
    d = False
    state = env.reset()
    # drop the score board
    state = state[21:]
    life = max_life

    # compress the original frame
    get_init_state(history, state)
    print(history.shape)
    # show the state
    # im=Image.fromarray(history[0])
    # im.show()