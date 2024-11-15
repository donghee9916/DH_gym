from Environment.env import Env

import numpy as np 
import os 
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
# from torch.utils.tensorboard import SummaryWriter

from model.dqn import DQN, ReplayMemory
from model.actorcritic import ActorCritic, Actor, Critic
from IPython import display

class Runner():
    def __init__(self):

        self.env = Env(18)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # matplotlib setting 
        self.is_ipython = 'inline' in matplotlib.get_backend()
        if self.is_ipython:
            from IPython import display

        plt.ion()

        """
        Env Setting 
        """
        self.n_actions = self.env.action_space.shape # 3,5
        self.state = self.env.reset()
        self.n_observations = len(self.state)


        self.params = dict()
        self.Model = "DQN"  ## AC, DQN, PPO 
        self.Model_logs = self.Model + "_decision"

        """
        PLOT VARIABLES 
        """
        self.episode_durations = []
        self.episode_rewards = [] 

        # self.writer = SummaryWriter(self.Model_logs)

        """
        DQN Setting 
        """
        self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
        self.params["DQN"] = dict()
        self.params["DQN"]["BATCH_SIZE"] = 128
        self.params["DQN"]["GAMMA"] = 0.99
        self.params["DQN"]["EPS_START"] = 0.9
        self.params["DQN"]["EPS_END"] = 0.05
        self.params["DQN"]["EPS_DECAY"] = 1000
        self.params["DQN"]["TAU"] = 0.005
        self.params["DQN"]["LR"] = 1e-4

        self.policy_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net = DQN(self.n_observations, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr = self.params["DQN"]["LR"], amsgrad=True)
        self.memory = ReplayMemory(10000)

        """
        Actor Critic Setting
        """
        self.params["AC"] = dict()
        self.params["AC"]["LR"] = 1e-4
        self.entropy = 0 
        

        self.plots = {"Actor Loss": [], "Critic Loss": [], "Reward": [], "Mean Reward": []}
        self.actor = Actor(self.n_observations, self.n_actions).to(self.device)
        self.critic = Critic(self.n_observations).to(self.device)
        self.AC = ActorCritic(self.n_observations, self.n_actions).to(self.device)

        self.optimizerA = optim.AdamW(self.actor.parameters(), lr = self.params["AC"]["LR"])
        self.optimizerC = optim.AdamW(self.critic.parameters(), lr= self.params["AC"]["LR"])
        
        self.steps_done = 0


    def save_model(self, dqn_policy_net, episode_num):
        if not os.path.exists('output'):
            os.makedirs('output')

        save_path = f'output/{self.Model}_model_episode_{episode_num}.pth'
        if self.Model == "DQN":
            torch.save(dqn_policy_net.state_dict(), save_path)
            print(f"Model saved to episode {episode_num} to {save_path}")


    def select_action(self, state):
        sample = random.random()
        eps_threhold = self.params["DQN"]["EPS_END"] + (self.params["DQN"]["EPS_START"] - self.params["DQN"]["EPS_END"])* \
            math.exp(-1. * self.steps_done / self.params["DQN"]["EPS_DECAY"])
        self.steps_done += 1

        if sample > eps_threhold:
            with torch.no_grad():
                action_values = self.policy_net(state)
                action_1 = torch.argmax(action_values[:, :3], dim=-1)  # 첫 번째 액션 공간
                action_2 = torch.argmax(action_values[:, 3:], dim=-1)  # 두 번째 액션 공간
                action = torch.stack([action_1, action_2], dim=1)
                return action
        else:
            return torch.tensor([[self.env.action_space.sample()[0], self.env.action_space.sample()[1]]], device=self.device, dtype=torch.long)


    def plot_durations(self,show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        reward_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Rewards')
        # plt.plot(durations_t.numpy())
        plt.plot(reward_t.numpy())
        plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if self.Model == "DQN":
            if len(self.memory) < self.params["DQN"]["BATCH_SIZE"]:
                return
            transitions = self.memory.sample(self.params["DQN"]["BATCH_SIZE"])
            batch = self.Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
            non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])

            print(f"state batch : {batch.state}, type : {type(batch.state)}\n")
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
            next_state_values = torch.zeros(self.params["DQN"]["BATCH_SIZE"], device=self.device)
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
            
    def estimate_value(self, state):
        """
        Actor Critic Estimation
        """
        pred = self.critic(state)
        if self.critic.value_history.dim() != 0 :
            self.critic.value_history = torch.cat([self.critic.value_history, pred])
        else:
            self.critic.value_history = (pred)

    def train(self):
        if torch.cuda.is_available():
            num_episodes = 2000000
        else:
            num_episodes = 50

        
        smoothed_reward = []
        for i_episode in range(num_episodes):
            self.env.episode_num = i_episode + 1 

            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            """
            Actor Critic
            """
            rewards = 0
            self.entropy = 0
            smooth = 10 
            done = False

            if self.Model == "DQN":
                for t in count():
                    action = self.select_action(state)
                    
                    # self.env.render()
                    # if i_episode % 10 == 0 :
                    #     self.env.render()
                    observation, reward, done = self.env.step(action.squeeze(0).cpu().numpy())
                    reward = torch.tensor([reward], device=self.device)
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    # print(f"state : {state}, type : {type(state)}")
                    # print(f"action : {action}, type : {type(action)}")
                    # print(f"next_state : {next_state}, type : {type(next_state)}")
                    # print(f"reward : {reward}, type : {type(reward)}\n")


                    self.memory.push(state, action, next_state, reward)
                    state = next_state
                    self.optimize_model()
                    target_net_state_dict = self.target_net.state_dict()
                    policy_net_state_dict = self.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.params["DQN"]["TAU"] + target_net_state_dict[key]*(1-self.params["DQN"]["TAU"])
                    self.target_net.load_state_dict(target_net_state_dict)
                    if done:
                        self.episode_durations.append(t + 1)
                        self.episode_rewards.append(reward.item())
                        self.plot_durations()
                        if (i_episode +1) % 500 == 0:
                            self.save_model(self.policy_net, i_episode + 1)
                        break

            elif self.Model == "AC":
                for t in count():
                    self.estimate_value(state)
                    policy = self.actor(state).cpu().detach().numpy()
                    action = self.select_action(state)

                    e = - np.sum(np.mean(policy) * np.log(policy))
                    self.entropy += e 

                    state, reward, done = self.env.step(action.squeeze(0).cpu().numpy())
                    rewards += reward

                    self.actor.reward_episode.append(reward)

                    if done:
                        self.episode_durations.append(t + 1)
                        self.episode_rewards.append(reward.item())
                        self.plot_durations()
                        # if (i_episode +1) % 500 == 0:
                        #     self.save_model(self.policy_net, i_episode + 1)
                        break
                smoothed_reward.append(rewards)
                if len(smoothed_reward) > smooth: 
                    smoothed_reward = smoothed_reward[-1*smooth: -1]

                a_loss, c_loss = self.update_a2c()
                
                # self.writer.add_scalar("Critic Loss", c_loss, i_episode)
                # self.writer.add_scalar("Actor Loss", a_loss, i_episode)
                # self.writer.add_scalar("Reward", rewards, i_episode)
                # self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), i_episode)

                self.plots["Critic Loss"].append(c_loss * 100)
                self.plots["Actor Loss"].append(a_loss)
                self.plots["Reward"].append(rewards)
                self.plots["Mean Reward"].append(np.mean(smoothed_reward))
            # if i_episode % 10 == 0 :
            #         self.env.close()
                    
                
        
        print('Complete')
        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()


def main():
    train_runner = Runner()
    train_runner.train()

if __name__ == "__main__":
    main()