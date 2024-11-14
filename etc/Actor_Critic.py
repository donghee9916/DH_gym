import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.distributions import Categorical
import ffmpeg

import numpy as np
import matplotlib.pyplot as plt 



import argparse
import seaborn as sns
import os

"""

"""
print("============================================================================================")
if (torch.cuda.is_available()):
  device = torch.device('cuda:0')
  torch.cuda.empty_cache()
  print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
  device = torch.device('cpu')
  print("Device set to : cpu")
print("============================================================================================")




# actor network
class Actor(nn.Module): 
    def __init__(self, in_size, out_size, hidden_units, dropout_rate): 
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(in_size, hidden_units).to(device)
        self.linear2 = nn.Linear(hidden_units, out_size).to(device)
        self.dropout = nn.Dropout(dropout_rate)
        self.softmax = nn.Softmax(dim=1)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.reward_episode = []

        self.reward_history = []
        self.loss_history = []
  
    def forward(self, x): 
        # convert numpy state to tensor
        # x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
        # x = Variable(x).to(device)
        x = torch.tensor(x).float().unsqueeze(0).to(device)
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.softmax(self.linear2(x))
        return x

# critic network
class Critic(nn.Module): 
    def __init__(self, in_size, hidden_units, dropout_rate): 
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(in_size, hidden_units).to(device)
        self.linear2 = nn.Linear(hidden_units, 1).to(device)
        self.dropout = nn.Dropout(dropout_rate)

        self.value_episode = []
        self.value_history = Variable(torch.Tensor()).to(device)
    
    def forward(self, x): 
        # x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
        # x = Variable(x).to(device)
        x = torch.tensor(x).float().unsqueeze(0).to(device)
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

class ActorCritic(nn.Module): 
    def __init__(self, actor, critic): 
        super(ActorCritic, self).__init__()
        self.actor = actor
        self.critic = critic
  
    def forward(self, x):
        value = self.critic(x)
        policy = self.actor(x)
        return value, policy

class Actor_Critic_Runner():
    def __init__ (self, actor, critic, a_optimizer, c_optimizer, gamma=0.99, entropy_coeff=0.001, value_loss_coeff=0.5, actor_loss_coeff=1.0, gradients_clipping=0.5, logs="a2c_cartpole"):
        self.actor = actor 
        self.critic = critic
        self.a_opt = a_optimizer
        self.c_opt = c_optimizer
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.value_loss_coeff = value_loss_coeff
        self.actor_loss_coeff = actor_loss_coeff
        self.gradients_clipping = gradients_clipping
        self.logs = logs 
        self.writter = SummaryWriter(logs)
        self.entropy = 0 
        self.plots = {
            "Actor Loss" : [],
            "Critic Loss" : [],
            "Reward" : [],
            "Mean Reward": []
        }
    
    def select_action(self, state):
        # convert state to tensor 
        probs = self.actor(state)
        print(state)
        c = Categorical(probs)
        action = c.sample()

        # place log probabilities into the policy history log\pi (a | s )
        if self.actor.policy_history.dim() != 0:
            self.actor.policy_history = torch.cat([self.actor.policy_history, c.log_prob(action)])
        else:
            self.actor.policy_history = (c.log_prob(action))

        return action
    
    def estimate_value(self, state):
        pred = self.critic(state).squeeze(0)
        if self.critic.value_history.dim() != 0:
            self.critic.value_history = torch.cat([self.critic.value_history, pred])
        else:
            self.critic.value_history = (pred)

    def calculate_reward(self, state, action, lap_time, collision_flag, prev_action):
        reward = 1 

        if state.dim() == 2:
            state_np = state.cpu().numpy() if state.is_cuda else state.numpy()
        else:
            # state가 1D 텐서인 경우 reshape 해서 사용
            state_np = state.unsqueeze(0).cpu().numpy() if state.is_cuda else state.unsqueeze(0).numpy()

        ego_velocity = state_np[0][0]  # 예를 들어 첫 번째 값
        ego_lane_order = state_np[0][1]
        gpp_lane_order = state_np[0][2]
        speed_limit = state_np[0][3]
        ego_prev_action = state_np[0][4]
        maximum_lane_num = state_np[0][5]

        if gpp_lane_order == 5:
            if action > 3:
                reward -= 1.0
        elif gpp_lane_order == 4:
            if action > 4:
                reward -= 1.0
        elif gpp_lane_order == 3:
            if action > 5 or action < 1:
                reward -= 1.0
        elif gpp_lane_order == 2:
            if action < 2:
                reward -= 1.0
        elif gpp_lane_order == 1:
            if action < 3:
                reward -= 1.0

        if action != prev_action:   ## 판단 떨림 방지
            reward -= 1.0


        if collision_flag:
            reward -= 1000.0

        return torch.FloatTensor([reward]).to(device)


    def update_a2c(self):
        R = 0 
        q_vals = []

        # "Unroll" thre rewards, apply gamma 
        for r in self.actor.reward_episode[::-1]:
            R = r + self.gamma * R 
            q_vals.insert(0,R)

        q_vals = torch.FloatTensor(q_vals).to(device)
        values = self.critic.value_history 
        log_probs = self.actor.policy_history 
    
        advantage = q_vals - values

        self.c_opt.zero_grad()
        critic_loss = self.value_loss_coeff * advantage.pow(2).mean()
        critic_loss.backward()
        self.c_opt.step()

        self.a_opt.zero_grad()
        actor_loss = (-log_probs * advantage.detach()).mean() + self.entropy_coeff* self.entropy
        actor_loss.backward()
        self.a_opt.step()

        self.actor.reward_episode = []
        self.actor.policy_history = Variable(torch.Tensor()).to(device)
        self.critic.value_history = Variable(torch.Tensor()).to(device)

        return actor_loss, critic_loss


    def save(self):
        ac = ActorCritic(self.actor, self.critic)
        torch.save(ac.state_dict(), '%s/model.pt'%self.logs)

    def plot(self):
        sns.set()
        sns.set_context("poster")
        plt.figure(figsize=(20, 16))
        plt.plot(np.arange(len(self.plots["Actor Loss"])), self.plots["Actor Loss"], label="Actor")
        plt.plot(np.arange(len(self.plots["Critic Loss"])), self.plots["Critic Loss"], label="Critic (x100)")
        plt.legend()
        plt.title("A2C Loss")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.savefig("%s/plot_%s.png"%(self.logs, "loss"))

        plt.figure(figsize=(20, 16))
        plt.plot(np.arange(len(self.plots["Reward"])), self.plots["Reward"], label="Reward")
        plt.plot(np.arange(len(self.plots["Mean Reward"])), self.plots["Mean Reward"], label="Mean Reward")
        plt.legend()
        plt.title("A2C Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.savefig("%s/plot_%s.png"%(self.logs, "rewards"))
