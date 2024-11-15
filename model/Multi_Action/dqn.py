import random
import numpy as np
import math 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple, deque
import os 

# Transition 정의 (Replay Memory에 저장될 항목)
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


class DQN_Base(nn.Module):
    def __init__(self, n_observations, action_shapes, params):
        super(DQN_Base, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 256)
        self.layer3 = nn.Linear(256, 128)
        self.params = params
        
        # 각 액션 공간에 대한 출력을 위한 두 개의 별도 레이어
        self.layer_action_1 = nn.Linear(128, 3)  # 첫 번째 액션 공간
        self.layer_action_2 = nn.Linear(128, 5)  # 두 번째 액션 공간

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        
        action_1 = self.layer_action_1(x)
        action_2 = self.layer_action_2(x)
        
        # 두 액션 공간을 이어서 출력
        action_values = torch.cat([action_1, action_2], dim=1)
        return action_values

class DQN(nn.Module):
    def __init__(self, n_observations, action_shapes, params):
        super(DQN, self).__init__()
        self.policy_net = DQN_Base(n_observations, action_shapes, params).to(device)
        self.target_net = DQN_Base(n_observations, action_shapes, params).to(device)
        self.params = params
        self.memory = ReplayMemory(10000)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.params["DQN"]["LR"], amsgrad=True)

        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

    def select_action(self, state, steps_done, env):
        sample = random.random()
        eps_threhold = self.params["DQN"]["EPS_END"] + (self.params["DQN"]["EPS_START"] - self.params["DQN"]["EPS_END"])* \
            math.exp(-1. * steps_done / self.params["DQN"]["EPS_DECAY"])
        steps_done += 1 

        if sample > eps_threhold:
            with torch.no_grad():
                action_values = self.policy_net(state)
                action_1 = torch.argmax(action_values[:, :3], dim=-1)
                action_2 = torch.argmax(action_values[:, 3:], dim=-1)
                action = torch.stack([action_1, action_2], dim=1)
                return action, steps_done
        else:
            return torch.tensor([[env.action_space.sample()[0], env.action_space.sample()[1]]], device=device, dtype=torch.long), steps_done

    def optimize_model(self):
        if len(self.memory) < self.params["DQN"]["BATCH_SIZE"]:
            return 
        transitions = self.memory.sample(self.params["DQN"]["BATCH_SIZE"])
        batch = self.Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s : s is not None,
                                      batch.next_state)),device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        state_action_values = self.policy_net(state_batch)

        action_1 = action_batch[:, 0]
        action_2 = action_batch[:, 1]
        state_action_values_1 = state_action_values[:, :3]  # 첫 번째 액션 공간
        state_action_values_2 = state_action_values[:, 3:]  # 두 번째 액션 공간
        selected_state_action_values = (
            state_action_values_1.gather(1, action_1.unsqueeze(1)) +
            state_action_values_2.gather(1, action_2.unsqueeze(1))
        ).squeeze(1)
        next_state_values = torch.zeros(self.params["DQN"]["BATCH_SIZE"], device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # 기대 Q 값 계산
        expected_state_action_values = (next_state_values * self.params["DQN"]["GAMMA"]) + reward_batch

        # Huber 손실 계산
        criterion = nn.SmoothL1Loss()
        loss = criterion(selected_state_action_values, expected_state_action_values)

        # 모델 최적화
        self.optimizer.zero_grad()
        loss.backward()
        # 변화도 클리핑 바꿔치기
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()
        

    def save_model(self, episode_num):
        """모델 저장 함수"""
        if not os.path.exists('output'):
            os.makedirs('output')

        save_path = f'output/dqn_model_episode_{episode_num}.pth'
        torch.save(self.policy_net.state_dict(), save_path)
        print(f"Model saved at episode {episode_num} to {save_path}")
