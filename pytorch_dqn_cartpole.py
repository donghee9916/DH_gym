import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from env import Env
import numpy as np
import os 


# env = gym.make("CartPole-v1")
env = Env(12)

# matplotlib 설정
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# GPU를 사용할 경우
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))

def save_model(policy_net, episode_num):
    # output 폴더가 없을 경우 생성
    if not os.path.exists('output'):
        os.makedirs('output')

    save_path = f'output/dqn_model_episode_{episode_num}.pth'

    torch.save(policy_net.state_dict(), save_path)
    print(f"Model saved at episode {episode_num} to {save_path}")


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
        # self.layer3 = nn.Linear(128, n_actions)
        
        self.layer_action_1 = nn.Linear(128,3)
        self.layer_action_2 = nn.Linear(128,5)
        
        # print("Shape Printer")
        # print(shape for shape in action_shapes)
        # self.layer3 = nn.ModuleList([nn.Linear(128, shape) for shape in action_shapes])


    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        # return self.layer3(x)
        # return torch.cat([layer(x).unsqueeze(1) for layer in self.layer3], dim=1)
        
        action_1 = self.layer_action_1(x)
        action_2 = self.layer_action_2(x)
        action_values = torch.cat([action_1, action_2], dim=1)
        
        return action_values
    

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4


n_actions = env.action_space.shape # ex) (3,5)
state = env.reset()
n_observations = len(state)


policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max (1)은 각 행의 가장 큰 열 값을 반환합니다.
            # 최대 결과의 두번째 열은 최대 요소의 주소값이므로,
            # 기대 보상이 더 큰 행동을 선택할 수 있습니다.
            action_values = policy_net(state)

            # 각 액션 공간에 대해 가장 큰 Q-value 선택
            action_1 = torch.argmax(action_values[:, :3], dim=-1)  # 첫 번째 액션 공간
            action_2 = torch.argmax(action_values[:, 3:], dim=-1)  # 두 번째 액션 공간

            # 선택된 액션을 [1, 2] 형태로 결합
            action = torch.stack([action_1, action_2], dim=1)

            return action  # [1, 1, n_actions] 형태로 반환
    else:
        return torch.tensor([[env.action_space.sample()[0], env.action_space.sample()[1]]], device=device, dtype=torch.long)


episode_durations = []
episode_rewards = [] 


def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    reward_t = torch.tensor(episode_rewards, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    # plt.plot(durations_t.numpy())
    plt.plot(reward_t.numpy())
    # 100개의 에피소드 평균을 가져 와서 도표 그리기
    # if len(durations_t) >= 100:
    #     # means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #     # means = torch.cat((torch.zeros(99), means))
    #     # plt.plot(means.numpy())
    #     means = reward_t.unfold(0, 100, 1).mean(1).view(-1)
    #     means = torch.cat((torch.zeros(99), means))
    #     plt.plot(means.numpy())
        

    plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # 최종이 아닌 상태의 마스크를 계산하고 배치 요소를 연결합니다
    # (최종 상태는 시뮬레이션이 종료 된 이후의 상태)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Q(s_t, a) 계산 - 모델이 Q(s_t)를 계산하고, 취한 행동의 열을 선택합니다.
    # 이들은 policy_net에 따라 각 배치 상태에 대해 선택된 행동입니다.
    # Q(s_t, a) 계산
    # state_action_values = policy_net(state_batch).gather(1, action_batch)
    state_action_values = policy_net(state_batch)

    # 각 배치 상태에서의 액션에 해당하는 Q값을 선택
    action_1 = action_batch[:, 0]
    action_2 = action_batch[:, 1]
    state_action_values_1 = state_action_values[:, :3]  # 첫 번째 액션 공간
    state_action_values_2 = state_action_values[:, 3:]  # 두 번째 액션 공간
    selected_state_action_values = (
        state_action_values_1.gather(1, action_1.unsqueeze(1)) +
        state_action_values_2.gather(1, action_2.unsqueeze(1))
    ).squeeze(1)

    # 모든 다음 상태를 위한 V(s_{t+1}) 계산
    # non_final_next_states의 행동들에 대한 기대값은 "이전" target_net을 기반으로 계산됩니다.
    # max(1).values로 최고의 보상을 선택하십시오.
    # 이것은 마스크를 기반으로 병합되어 기대 상태 값을 갖거나 상태가 최종인 경우 0을 갖습니다.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # 기대 Q 값 계산
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    

    # Huber 손실 계산
    criterion = nn.SmoothL1Loss()
    loss = criterion(selected_state_action_values, expected_state_action_values)

    # 모델 최적화
    optimizer.zero_grad()
    loss.backward()
    # 변화도 클리핑 바꿔치기
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

if torch.cuda.is_available():
    num_episodes = 2000
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # 환경과 상태 초기화
    # state : [-0.02880219  0.04266532  0.00012848 -0.04543126], type : <class 'numpy.ndarray'>
    # info : {}, type : <class 'dict'>
    # state, info = env.reset()
    state = env.reset()

    # state : tensor([[-7.0209e-05,  2.7393e-02,  4.7151e-02,  4.1970e-02]], device='cuda:0'), type : <class 'torch.Tensor'
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        # action : tensor([[0]], device='cuda:0'), type : <class 'torch.Tensor'>
        action = select_action(state)



        observation, reward, done = env.step(action.squeeze(0).cpu().numpy())  # .squeeze()로 텐서의 차원을 제거
        # if i_episode // 10 == 0 : 
        #     env.visualize(real_time=True) 
        reward = torch.tensor([reward], device=device)
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

        # 메모리에 변이 저장
        memory.push(state, action, next_state, reward)

        # 다음 상태로 이동
        state = next_state

        # (정책 네트워크에서) 최적화 한단계 수행
        optimize_model()

        # 목표 네트워크의 가중치를 소프트 업데이트
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(reward.item())
            plot_durations()

            if (i_episode +1) % 500 == 0:
                save_model(policy_net, i_episode + 1)

            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()