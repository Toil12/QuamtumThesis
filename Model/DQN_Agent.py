from .DQN_Classical import DQN
from .DQN_Q import *
from torch import nn
import random
import torch
import numpy as np
from collections import deque
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from Train.read_write_operations import ProjectIO
np.set_printoptions(threshold=np.inf)
# it uses Neural Network to approximate q function
# and replay memory & target q network

class DQNAgent():
    def __init__(self,
                 action_size:int,
                 io_obj:ProjectIO,
                 config_name:str,
                 strategy:str='c',
                 load_model:str="",
                 encode_mode:str="normal_a"):
        # if you want to see Cartpole learning, then change to True
        if torch.cuda.is_available():
            self.train_device = "gpu"
        else:
            self.train_device = "cpu"
        self.render = False
        self.load_model = False
        self.encode_mode=io_obj.encode_mode_dict[encode_mode]

        # get size of action
        self.action_size = action_size

        # read parameters from json file
        # parameters=read_parameters("train_lightning")
        parameters=io_obj.read_parameters()

        # These are hyper parameters for the DQN
        self.history_size=4
        self.discount_factor = parameters['discount_factor']
        self.learning_rate = parameters['learning_rate']
        self.memory_size = parameters['memory_size']
        self.epsilon = parameters['epsilon']
        self.epsilon_min = parameters['epsilon_min']
        self.explore_step = parameters['explore_step']
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = parameters['batch_size']
        self.train_start = parameters['train_start']
        self.update_target = parameters['update_target']
        self.q_device=parameters['device_name']

        self.n_layers=parameters['n_layers']
        self.re_upload=parameters['re_upload']

        # create replay memory using deque
        self.memory = deque(maxlen=self.memory_size)

        # create main model and target model with different strategy

        if strategy=="c":
            self.model = DQN(action_size)
            self.target_model = DQN(action_size)
        elif strategy=="q":
            q_part=DQN_Q(a_size=3,
                         n_layers=self.n_layers,
                         w_input=False,
                         w_output=False,
                         data_reupload=self.re_upload,
                         device_name=self.q_device,
                         encode_mode=self.encode_mode)
            cnn_outfeature=0
            if self.encode_mode==0:
                cnn_outfeature=4
            elif self.encode_mode==1:
                cnn_outfeature=8
            self.model = nn.Sequential(CNN_Compress(cnn_outfeature),q_part)
            self.target_model = nn.Sequential(CNN_Compress(cnn_outfeature),q_part)

        if self.train_device=="gpu":
            self.model.gpu()
            self.target_model.gpu()
        else:
            self.model.cpu()
            self.target_model.cpu()
        self.model.apply(self.weights_init)


        # self.optimizer = optim.RMSprop(params=self.model.parameters(),lr=self.learning_rate, eps=0.01, momentum=0.95)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

        if load_model!="":
            self.model = torch.load(f'SaveModels/{load_model}')

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            # print(m)
        elif classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            # print(m)

    # after some time interval update the target model to be same with model
    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    #  get action from model using epsilon-greedy policy
    #  action include 3,not move, to left or right, 1,2,3 in sample space
    def get_action(self, state):

        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:

            state = torch.from_numpy(state).unsqueeze(0)
            if self.train_device=="gpu":
                tate = Variable(state).float().gpu()
                action = self.model(state).data.gpu().max(1)[1]
            else:
                state = Variable(state).float().cpu()
                action = self.model(state).data.cpu().max(1)[1]
            return int(action)

    # save sample <s,a,r,s'> to the replay memory
    def append_sample(self, history, action, reward, done):
        self.memory.append((history, action, reward, done))

    def get_sample(self, frame):
        mini_batch = []
        if frame >= self.memory_size:
            sample_range = self.memory_size
        else:
            sample_range = frame

        # history size
        sample_range -= (self.history_size + 1)

        idx_sample = random.sample(range(sample_range), self.batch_size)
        for i in idx_sample:
            sample = []
            for j in range(self.history_size + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample)
            mini_batch.append((np.stack(sample[:, 0], axis=0), sample[3, 1], sample[3, 2], sample[3, 3]))

        return mini_batch

    # pick samples randomly from replay memory (with batch_size)
    def train_model(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.get_sample(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        # print(mini_batch)
        # 0,1,2,3:image,actions sequence,rewards,done

        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3]

        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.Tensor(states)
        if self.train_device=="gpu":
            states = Variable(states).float().gpu()
        else:
            states = Variable(states).float().cpu()

        pred = self.model(states)

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)
        # quantum output problem
        # print(one_hot_action.shape)
        # print(pred.shape)
        if self.train_device=="gpu":
            pred = torch.sum(pred.mul(Variable(one_hot_action).gpu()), dim=1)
        else:
            pred = torch.sum(pred.mul(Variable(one_hot_action).cpu()), dim=1)

        # Q function of next state
        next_states = torch.Tensor(next_states)
        if self.train_device=="gpu":
            next_states = Variable(next_states).float().gpu()
            next_pred = self.target_model(next_states).data.gpu()
        else:
            next_states = Variable(next_states).float().cpu()
            next_pred = self.target_model(next_states).data.cpu()

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        if self.train_device=="gpu":
            target = Variable(target).gpu()
        else:
            target = Variable(target).cpu()

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = F.smooth_l1_loss(pred, target)
        # print("loss check")
        loss.backward()

        # and train
        self.optimizer.step()