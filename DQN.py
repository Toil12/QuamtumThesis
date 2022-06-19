import sys
import gym
import torch
import numpy as np
from collections import deque
from copy import deepcopy
from skimage.transform import resize
from skimage.color import rgb2gray
from Model.DQN_Agent import DQNAgent
from Train.read_write_operations import ProjectIO
import time
import matplotlib.pyplot as plt
import sys,getopt

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

if __name__ == "__main__":
    EPISODES = 500000
    HEIGHT = 84
    WIDTH = 84
    HISTORY_SIZE = 4
    #
    input_list=get_input_parameters()
    print(input_list)
    model_type = input_list[0]
    config_name = input_list[1]
    encode_mode = input_list[2]
    render_mode = None
    # input parameters
    # render_mode='humam'
    #

    io_obj = ProjectIO(model_type=model_type,
                       encod_mode=encode_mode)
    image_title=io_obj.image_title
    # title = get_image_title(model_type, io_obj.encode_mode_dict[encode_mode])
    #
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

    agent = DQNAgent(action_size=action_size,
                     io_obj=io_obj,
                     config_name=config_name,
                     strategy=model_type,
                     encode_mode=io_obj.encode_mode)
    recent_reward = deque(maxlen=100)
    frame = 0
    start_train_tag=False
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
                # print("start one train at frame ", frame)
                if not start_train_tag:
                    print("start one train at frame ",frame)
                    start_train_tag=True
                else:
                    pass
                agent.train_model(frame)
                # print("check")
                if frame % agent.update_target == 0:
                    agent.update_target_model()
            score += reward
            # shift one step at tail
            history[:4, :, :] = history[1:, :, :]
            time_end=time.time()
            # plot intermediate result
            if frame % 500 == 0:
                # print('now time : ', datetime.now())
                scores.append(score)
                episodes.append(int(e))
                # print(episodes,scores)
                plt.plot(episodes,scores)
                # plt.title(f"{image_title}")
                plt.xlabel("Episodes")
                plt.ylabel("Scores")
                plt.savefig(f"Results/Images/{io_obj.image_name}")
                plt.clf()

            if done and frame>=agent.train_start:
                recent_reward.append(score)
                # every episode, plot the play time
                print("episode:", e, "  score:", score, "  memory length:",
                      len(agent.memory), "  epsilon:", agent.epsilon, "   steps:", step,
                      "    recent reward:", np.mean(recent_reward),
                      "    time consume:",time_end-time_start)
                print("consumed frames ",frame)
                # write into log fiel

                log=f"episode:{e},score:{score},memory length:{len(agent.memory)},epsilon:{agent.epsilon}," \
                    f"steps:{step},recent reward:{np.mean(recent_reward)},time consume:{time_end-time_start}," \
                    f"frames:{frame}"
                io_obj.write_log(log)

                # if the mean of scores of last 10 episode is bigger than 400
                # stop training
                if np.mean(recent_reward) > 50:
                    torch.save(agent.model, f"SaveModels/{io_obj.model_name}")
                    sys.exit()