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

def encode():
    n_qubits=3
    dev=qml.device("lightning.qubit",wires=n_qubits)

    results=[]
    @qml.qnode(dev,interface='torch',diff_method="parameter-shift")
    def circuit(inputs):
        for x in range(inputs.shape[1]):
            for y in range(inputs.shape[2]):
                for i in range(4):
                    color=inputs[i][x][y]
                    trans_color(color)
                    if i==3:
                        trans_position(x,y)
        return [qml.expval(qml.PauliZ(wire)) for wire in range(n_qubits)]
    return circuit

def trans_color(color):
    qml.RX(color/(255*np.pi),2)

def trans_position(x,y):
    qml.RY(x / (2*np.pi),wires=0)
    qml.CNOT(wires=[0,2])
    qml.RZ(y / (2*np.pi),wires=1)
    qml.CNOT(wires=[1,2])


if __name__ == '__main__':
    EPISODES = 500000
    HEIGHT = 3
    WIDTH = 3
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
    for e in range(4):
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

        k=encode()
        start=time.time()
        r=k(history)
        end=time.time()
        print(end-start)
        print(r.shape)
        print(r)