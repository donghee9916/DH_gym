import gym
import torch
import torch.nn as nn
import numpy as np
from collections import deque
import random
from itertools import count
import torch.nn.functional as F
from tensorboardX import SummaryWriter


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    def __init__(self, n_observations, action_shapes):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(n_observations, 64)
        self.relu = nn.ReLU()
        self.fc_value = nn.Linear(64,256)
        self.fc_adv_1 = nn.Linear(64,256)
        self.fc_adv_2 = nn.Linear(64,256)

        self.value = nn.Linear(256,1)

        self.adv_1 = nn.Linear(256,action_shapes[0])
        self.adv_2 = nn.Linear(256, action_shapes[1])

    def forward(self, state):
        y = self.relu(self.fc1(state))
        value = self.relu(self.fc_value(y))
        adv_1 = self.relu(self.fc_adv_1(y))
        adv_2 = self.relu(self.fc_adv_2(y))

        value = self.value(value)
        adv_1 = self.adv_1(adv_1)
        adv_2 = self.adv_2(adv_2)

        advAverage_1 = torch.mean(adv_1, dim=1, keepdim=True)
        advAverage_2 = torch.mean(adv_2, dim=1, keepdim=True)
        Q1 = value + adv_1 - advAverage_1
        Q2 = value + adv_2 - advAverage_2
        return Q1, Q2

    def select_action(self, state):
        with torch.no_grad():
            Q1, Q2 = self.forward(state)
            action_1_index = torch.argmax(Q1, dim=1)
            action_2_index = torch.argmax(Q2, dim=1)
            
        return action_1_index.item(), action_2_index.item()


class DDQN_Memory(object):
    def __init__(self, memory_size: int) -> None:
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()


class DDQN(nn.Module):
    def __init__(self,n_observations, action_shapes, params):
        super(DDQN, self).__init__()
        self.params = params
        self.GAMMA = self.params["DDQN"]["GAMMA"]
        self.EXPLORE = self.params["DDQN"]["EXPLORE"]
        self.INITIAL_EPSILON = self.params["DDQN"]["INITIAL_EPSILON"]
        self.FINAL_EPSILON = self.params["DDQN"]["FINAL_EPSILON"]
        self.REPLAY_MEMORY = self.params["DDQN"]["REPLAY_MEMORY"]
        self.BATCH = self.params["DDQN"]["BATCH"]
        self.UPDATE_STEPS = self.params["DDQN"]["UPDATE_STEPS"]

        self.onlineQNetwork = QNetwork(n_observations, action_shapes).to(device)
        self.targetQNetwork = QNetwork(n_observations, action_shapes).to(device)
        self.targetQNetwork.load_state_dict(self.onlineQNetwork.state_dict())
        self.optimizer = torch.optim.Adam(self.onlineQNetwork.parameters(), lr=1e-4)

        self.begin_learn = False
        self.memory = DDQN_Memory(self.REPLAY_MEMORY)

        self.epsilon = self.INITIAL_EPSILON
        self.learn_steps = 0 
        self.writer = SummaryWriter('logs/ddqn')

    def init(self):
        self.learn_steps = 0
        self.epsilon = self.INITIAL_EPSILON

    def select_action(self, state):
        # state should be float tensor
        action_1, action_2 = self.onlineQNetwork.select_action(state)
        return [action_1, action_2]
        
