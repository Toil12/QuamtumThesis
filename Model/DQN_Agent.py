class DQNAgent():
    def __init__(self, action_size:int,strategy:str):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False

        # get size of action
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.learning_rate = 0.0001
        self.memory_size = 1000000
        self.epsilon = 1.0
        self.epsilon_min = 0.02
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.batch_size = 32
        self.train_start = 100
        self.update_target = 1000

        # create replay memory using deque
        self.memory = deque(maxlen=self.memory_size)

        # create main model and target model

        if strategy=="c":
            self.model = DQN(action_size)
            self.target_model = DQN(action_size)
        elif strategy=="q":
            self.model = DQN_Q(action_size)
            self.target_model = DQN_Q(action_size)

        self.model.cuda()
        self.model.apply(self.weights_init)
        self.target_model.cuda()

        # self.optimizer = optim.RMSprop(params=self.model.parameters(),lr=self.learning_rate, eps=0.01, momentum=0.95)
        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

        # initialize target model
        self.update_target_model()

        if self.load_model:
            self.model = torch.load('save_model/breakout_dqn')

    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            print(m)
        elif classname.find('Conv') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            print(m)

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
            state = Variable(state).float().cuda()
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
        sample_range -= (HISTORY_SIZE + 1)

        idx_sample = random.sample(range(sample_range), self.batch_size)
        for i in idx_sample:
            sample = []
            for j in range(HISTORY_SIZE + 1):
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
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3]

        # bool to binary
        dones = dones.astype(int)

        # Q function of current state
        states = torch.Tensor(states)
        states = Variable(states).float().cuda()
        pred = self.model(states)

        # one-hot encoding
        a = torch.LongTensor(actions).view(-1, 1)

        one_hot_action = torch.FloatTensor(self.batch_size, self.action_size).zero_()
        one_hot_action.scatter_(1, a, 1)

        pred = torch.sum(pred.mul(Variable(one_hot_action).cuda()), dim=1)

        # Q function of next state
        next_states = torch.Tensor(next_states)
        next_states = Variable(next_states).float().cuda()
        next_pred = self.target_model(next_states).data.cpu()

        rewards = torch.FloatTensor(rewards)
        dones = torch.FloatTensor(dones)

        # Q Learning: get maximum Q value at s' from target model
        target = rewards + (1 - dones) * self.discount_factor * next_pred.max(1)[0]
        target = Variable(target).cuda()

        self.optimizer.zero_grad()

        # MSE Loss function
        loss = F.smooth_l1_loss(pred, target)
        loss.backward()

        # and train
        self.optimizer.step()