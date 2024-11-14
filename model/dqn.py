import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
import os 

# Transition 정의 (Replay Memory에 저장될 항목)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, action_shapes):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        
        # 각 액션 공간에 대한 출력을 위한 두 개의 별도 레이어
        self.layer_action_1 = nn.Linear(128, 3)  # 첫 번째 액션 공간
        self.layer_action_2 = nn.Linear(128, 5)  # 두 번째 액션 공간

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        action_1 = self.layer_action_1(x)
        action_2 = self.layer_action_2(x)
        
        # 두 액션 공간을 이어서 출력
        action_values = torch.cat([action_1, action_2], dim=1)
        return action_values


def save_model(policy_net, episode_num):
    """모델 저장 함수"""
    if not os.path.exists('output'):
        os.makedirs('output')

    save_path = f'output/dqn_model_episode_{episode_num}.pth'
    torch.save(policy_net.state_dict(), save_path)
    print(f"Model saved at episode {episode_num} to {save_path}")
