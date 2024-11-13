import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import random
from collections import namedtuple, deque

import time 
import math

# Replay Memory 클래스 정의
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """transition을 메모리에 저장"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Q-network 정의
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr, gamma, epsilon, target_update, memory_capacity, batch_size):
        
        
        
        
        self.state_dim = state_dim      # 
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.target_update = target_update
        self.batch_size = batch_size
        self.memory = ReplayMemory(memory_capacity)
        self.prev_action = 0

        ################################## set device ##################################
        print("============================================================================================")
        # set device to cpu or cuda
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=lr)
        
        self.steps_done = 0

    def select_action(self, state):
        sample = random.random()
        # self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
        #                 math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        self.steps_done += 1
        if sample > self.epsilon:
            with torch.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.action_dim)]], device=self.device, dtype=torch.long)
    
    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        # # 현재 메모리에 쌓인 Transition 개수 출력
        # print(f"Memory Len : {len(self.memory)}")

        # 메모리에서 batch_size 크기만큼의 Transition 무작위 샘플링
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 다음 상태가 None이 아닌 것들에 대한 마스크 생성
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)
        # 다음 상태 중 None이 아닌 것들만을 선택하고, 이를 하나의 Tensor로 결합
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        # 배치에서 상태, 행동, 보상에 해당하는 부분을 Tensor로 결합
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # 현재 상태에서의 Q값 예측 (정책 네트워크 사용), 수행한 action에 해당하는 Q값 선택
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # 다음 상태의 Q값 초기화
        next_state_values = torch.zeros(self.batch_size, device=self.device)

        # 다음 상태가 존재하는 경우에만 target_net을 통해 Q값 예측, detach()로 그래프 분리
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()

        # 기대되는 Q값 계산: 보상 + (감마 * 다음 상태의 Q값)
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # 손실 함수 계산 (smooth_l1_loss는 Huber 손실)
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        
        # 역전파를 통해 네트워크 가중치 업데이트
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 일정 스텝마다 target_net을 policy_net의 가중치로 업데이트
        if self.steps_done % self.target_update == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())


    def calculate_reward(self, state, action, lap_time, collision_flag, prev_action):
        reward = 1.0
        
        ego_velocity = state[0]
        ego_lane_order = state[1]
        gpp_lane_order = state[2]
        speed_limit = state[3]
        ego_prev_action = state[4]


        maximum_lane_num = state[5]

        if gpp_lane_order == 5:
            if action >3:
                reward -= 2.0
        elif gpp_lane_order == 4:
            if action >4:
                reward -= 2.0
        elif gpp_lane_order == 3:
            if action >5 or action <1:
                reward -= 2.0
        elif gpp_lane_order == 2:
            if action < 2:
                reward -= 2.0
        elif gpp_lane_order == 1:
            if action < 3:
                reward -= 2.0

        if action != prev_action:   ## 판단 떨림 방지
            reward -= 6.0
            print("Oss")

        if collision_flag:
            reward -= 100.0

        reward = torch.tensor([reward], dtype=torch.float32, device=self.device)

        return reward
