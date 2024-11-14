import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import time
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.distributions import Categorical
import ffmpeg

import argparse
import seaborn as sns
import os

from env import Env
from Actor_Critic import ActorCritic, Actor, Critic
state_dim = 12 
print("============================================================================================")
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


env= Env(state_dim)

class Runner():
    def __init__(self, actor, critic, a_optimizer, c_optimizer, gamma=0.99, logs = "a2c_cartpole"):
        self.actor = actor
        self.critic = critic
        self.a_opt = a_optimizer
        self.c_opt = c_optimizer
        self.gamma = gamma
        self.logs = logs
        self.writer = SummaryWriter(logs)
        self.entropy = 0
        self.plots = {"Actor Loss": [], "Critic Loss": [], "Reward": [], "Mean Reward": []}

    def env_step(self, action):
        state, reward, done = env.step(action)
        return torch.FloatTensor([state]).to(device), torch.FloatTensor([reward]).to(device), done
    
    def select_action(self, state):
        probs = self.actor(state)
        c = Categorical(probs)
        action = c.sample()

        #place log probabilities into the policy history log\pi(a | s)
        if self.actor.policy_history.dim()!= 0: 
            self.actor.policy_history = torch.cat([self.actor.policy_history, c.log_prob(action)])
        else: 
            self.actor.policy_history = (c.log_prob(action))
        
        return action
    def estimate_value(self, state): 
        pred = self.critic(state).squeeze(0)
        if self.critic.value_history.dim()!= 0: 
            self.critic.value_history = torch.cat([self.critic.value_history, pred])
        else: 
            self.critic.policy_history = (pred)
    def update_a2c(self):
        R = 0
        q_vals = []

        #"unroll" the rewards, apply gamma
        for r in self.actor.reward_episode[::-1]: 
            R = r + self.gamma * R
            q_vals.insert(0, R)
        
        q_vals = torch.FloatTensor(q_vals).to(device)
        values = self.critic.value_history
        log_probs = self.actor.policy_history
        
        # print(values)
        # print(log_probs)
        advantage = q_vals - values
    
        print(f"advantage : {advantage}")
        self.c_opt.zero_grad()
        critic_loss = 0.0005 * advantage.pow(2).mean()
        critic_loss.backward()
        self.c_opt.step()

        self.a_opt.zero_grad()
        actor_loss = (-log_probs * advantage.detach()).mean() + 0.001 * self.entropy
        actor_loss.backward()
        self.a_opt.step()

        self.actor.reward_episode = []
        self.actor.policy_history = Variable(torch.Tensor()).to(device)
        self.critic.value_history = Variable(torch.Tensor()).to(device)
        
    
        return actor_loss, critic_loss
    
    def train(self, episodes = 200, smooth = 10):
        smoothed_reward = []
        
        for episode in range(episodes): 
            rewards = 0
            state = env.reset()
            self.entropy = 0
            done = False

            while not done:
                self.estimate_value(state)
                policy = self.actor(state).cpu().detach().numpy()
                action = self.select_action(state)

                e = -np.sum(np.mean(policy) * np.log(policy))
                self.entropy += e

                state, reward, done = env.step(action.data[0].item())
                rewards+= reward


                self.actor.reward_episode.append(reward)

                if done:
                    break
        
            smoothed_reward.append(rewards)
            if len(smoothed_reward) > smooth: 
                smoothed_reward = smoothed_reward[-1*smooth: -1]

            a_loss, c_loss = self.update_a2c()
            
            print(a_loss)
            self.writer.add_scalar("Critic Loss", c_loss, episode)
            self.writer.add_scalar("Actor Loss", a_loss, episode)
            self.writer.add_scalar("Reward", rewards, episode)
            self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), episode)

            # self.plots["Critic Loss"].append(c_loss * 100)
            # self.plots["Actor Loss"].append(a_loss)
            # self.plots["Reward"].append(rewards)
            # self.plots["Mean Reward"].append(np.mean(smoothed_reward))
            self.plots["Actor Loss"].append(a_loss.cpu().detach().numpy())
            self.plots["Critic Loss"].append(c_loss.cpu().detach().numpy())
            self.plots["Reward"].append(np.sum(self.actor.reward_episode))
            # self.plots["Mean Reward"].append(np.mean(self.plots["Reward"][-100:]))
            self.plots["Mean Reward"].append(np.mean(smoothed_reward))

            if episode % 20 == 0: 
                print("\tEpisode {} \t Final Reward {:.2f} \t Average Reward: {:.2f}".format(episode, rewards, np.mean(smoothed_reward)))

    def run(self):
        sns.set_style("dark")
        sns.set_context("poster")

        fig = plt.figure() 
        ims = []
        rewards = 0
        state = env.reset()
        for time in range(500):
            action = self.select_action(state) 
            state, reward, done = env.step(action.data[0].item())
            rewards += reward

            if done:
                break
        
            im = plt.imshow(env.render(mode='rgb_array'), animated=True)
            plt.axis('off')
            plt.title("Actor Critic Agent")
            ims.append([im])

        print("\tTotal Reward: ", rewards)
        env.close()
        print("\tSaving Animation ...")
        ani = animation.ArtistAnimation(fig, ims, interval=20, blit=True,
                                        repeat_delay=1000)
        ani.save('%s-movie.avi'%self.logs, dpi = 300)
    
    def save(self): 
        ac = ActorCritic(self.actor, self.critic)
        torch.save(ac.state_dict(),'%s/model.pt'%self.logs)

    def plot(self):
        sns.set()
        sns.set_context("poster")

        plt.figure(figsize=(20, 16))
        plt.plot(np.arange(len(self.plots["Actor Loss"])), self.plots["Actor Loss"], label = "Actor")
        plt.plot(np.arange(len(self.plots["Critic Loss"])), self.plots["Critic Loss"], label = "Critic (x100)")
        #  # Actor Loss는 텐서일 수 있으므로 .cpu().numpy()로 변환
        # plt.plot(np.arange(len(self.plots["Actor Loss"])), 
        #         np.array(self.plots["Actor Loss"]).cpu().numpy(), label="Actor")
        # # Critic Loss도 마찬가지로 .cpu().numpy()로 변환
        # plt.plot(np.arange(len(self.plots["Critic Loss"])), 
        #         np.array(self.plots["Critic Loss"]).cpu().numpy(), label="Critic (x100)")
        
        plt.legend()
        plt.title("A2C Loss")
        plt.xlabel("Episodes")
        plt.ylabel("Loss")
        plt.show()
        plt.savefig("%s/plot_%s.png"%(self.logs, "loss"))


        plt.figure(figsize=(20, 16))
        plt.plot(np.arange(len(self.plots["Reward"])), self.plots["Reward"], label="Reward")
        plt.plot(np.arange(len(self.plots["Mean Reward"])), self.plots["Mean Reward"], label = "Mean Reward")
        plt.legend()
        plt.title("A2C Rewards")
        plt.xlabel("Episodes")
        plt.ylabel("Rewards")
        plt.show()
        plt.savefig("%s/plot_%s.png"%(self.logs, "rewards"))

def main(): 

    state_dim = 12
    action_dim =  3
    """
    Actor Critic Setting 
    """
    lr = 0.001                  # 학습 속도 
    gamma = 0.99                # 장기 보상에 대한 중요도 
    number_of_episodes = 2000   # Episode 의 수 
    batch_size = 64             # 학습 시 한 번에 사용할 경험의 수 
    hidden_units = 128          # Hidden layer size for both actor and critic networks
    entropy_coeff = 0.001       # 정책의 무작위성을 위한 value
    value_loss_coeff = 0.5      # Critic 네트워크의 가치 손실 가중치 --> 클수록 가치 추정이 더 중요하게 다뤄짐
    actor_loss_coeff = 1.0      # Actor 네트워크 정책 손실에 대한 가중치 --> 클수록 정책이 더 중요하게 다뤄짐
    dropout_rate = 0.2          # 신경망 Dropout 비율  --> 과적합을 방지하기 위함으로 특정 뉴런 학습 중 랜덤하게 끄는 비율 
    Replay_Buffer_size = 2000   # 경험을 저장하는 메커니즘으로 주로 Off-Policy에 사용된다. A2C는 사용되지 않지만 A3C(Asynchronous Advantage Actor-Critic)과 같은 변형에 사용 
    gradients_clipping = 0.5    # 기울기가 너무 커지는 것을 방지 
    
    actor = Actor(state_dim, action_dim, hidden_units, dropout_rate).to(device)
    critic = Critic(state_dim, hidden_units, dropout_rate).to(device)
    ac = ActorCritic(actor, critic)


    
    # #if we're loading a model
    # if args.load: 
    #     ac.load_state_dict(torch.load(args.model))
    #     actor = ac.actor
    #     critic = ac.critic

    a_optimizer = optim.Adam(actor.parameters(), lr = lr)
    c_optimizer = optim.Adam(critic.parameters(), lr = lr)

    runner = Runner(actor, critic, a_optimizer, c_optimizer, logs = "a2c_cartpole/%s" %time.time())
    
    episode_num = 200
    plot_flag = True
    save_flag = True
    run_flag = True

  
    runner.train(episode_num)

    if plot_flag:
        print("[Plot]\tPlotting Training Curves ...")
        runner.plot()

    if save_flag: 
        print("[Save]\tSaving Model ...")
        runner.save()

    # if run_flag:
    #     print("[Run]\tRunning Simulation ...")
    #     runner.run()
    

    print("[End]\tDone. Congratulations!")

if __name__ == '__main__':
    main()