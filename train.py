from Environment.env import Env

import numpy as np 
import os 
import math
import random
import queue
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count



import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F 
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp

from model.Multi_Action.dqn import DQN_Base, DQN, ReplayMemory
from model.actorcritic import ActorCritic, Actor, Critic
from model.Multi_Action.ppo import * 
from model.Multi_Action.discrete_A3C import *
from model.Multi_Action.ddqn import * 
from shared_adam import SharedAdam
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
        self.n_actions = self.env.action_space.nvec # 3,5
        self.state = self.env.reset()
        self.n_observations = len(self.state)


        self.params = dict()
        self.Model = "DQN"  ## AC, DQN, PPO, A3C, DDQN
        self.Model_logs = "logs/" + self.Model + "_decision"

        """
        PLOT VARIABLES 
        """
        self.episode_durations = []
        self.episode_rewards = [] 

        # self.writer = SummaryWriter(self.Model_logs)

        """
        DQN Setting 
        """
        if self.Model == "DQN":
        # self.Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward'))
            self.params["DQN"] = dict()
            self.params["DQN"]["BATCH_SIZE"] = 128
            self.params["DQN"]["GAMMA"] = 0.99
            self.params["DQN"]["EPS_START"] = 0.9
            self.params["DQN"]["EPS_END"] = 0.05
            self.params["DQN"]["EPS_DECAY"] = 1000
            self.params["DQN"]["TAU"] = 0.005
            self.params["DQN"]["LR"] = 1e-4

            # self.policy_net = DQN_Base(self.n_observations, self.n_actions, self.params).to(self.device)
            # self.target_net = DQN_Base(self.n_observations, self.n_actions, self.params).to(self.device)
            self.dqn = DQN(self.n_observations, self.n_actions, self.params)
            
            self.dqn.target_net.load_state_dict(self.dqn.policy_net.state_dict())

            # self.optimizer = optim.AdamW(self.dqn.policy_net.parameters(), lr = self.params["DQN"]["LR"], amsgrad=True)
            # self.memory = ReplayMemory(10000)
            self.steps_done = 0
        elif self.Model == "DDQN":
            self.params["DDQN"] = dict()
            self.params["DDQN"]["GAMMA"] = 0.99
            self.params["DDQN"]["EXPLORE"] = 20000
            self.params["DDQN"]["INITIAL_EPSILON"] = 0.1
            self.params["DDQN"]["FINAL_EPSILON"] = 0.0001
            self.params["DDQN"]["REPLAY_MEMORY"] = 50000
            self.params["DDQN"]["BATCH"] = 16
            self.params["DDQN"]["UPDATE_STEPS"] = 4


            self.ddqn = DDQN(self.n_observations, self.n_actions, self.params)

        # """
        # Actor Critic Setting
        # """
        # self.params["AC"] = dict()
        # self.params["AC"]["LR"] = 1e-4
        # self.entropy = 0 
        

        # self.plots = {"Actor Loss": [], "Critic Loss": [], "Reward": [], "Mean Reward": []}
        # self.actor = Actor(self.n_observations, self.n_actions).to(self.device)
        # self.critic = Critic(self.n_observations).to(self.device)
        # self.AC = ActorCritic(self.n_observations, self.n_actions).to(self.device)

        # self.optimizerA = optim.AdamW(self.actor.parameters(), lr = self.params["AC"]["LR"])
        # self.optimizerC = optim.AdamW(self.critic.parameters(), lr= self.params["AC"]["LR"])
        
        elif self.Model == "PPO":
            """
            Proximal Policy Optimization 
            """
            self.params["PPO"] = dict()
            self.params["PPO"]["lr_actor"] = 0.0003
            self.params["PPO"]["lr_critic"] = 0.001
            self.params["PPO"]["gamma"] = 0.99
            self.params["PPO"]["K_epochs"] = 80
            self.params["PPO"]["eps_clip"] = 0.20

            # if Continuous
            self.params["PPO"]["random_seed"] = 0                   
            self.params["PPO"]["action_std"] = 0.6                  # starting std for action distribution (Multivariate Normal
            self.params["PPO"]["action_decay_rate"] = 0.05          # linearly decay action_std (action_std  action_std - action_std_decay_rate)
            self.params["PPO"]["min_action_std"] = 0.1              # minimum action_std (stop decay after action_std = min_action_std) 
            self.params["PPO"]["action_std_decay_freq"] = 100000    # action_std decay frequency (in num timesteps) 

            print(self.n_actions)
            self.ppo = PPO(self.n_observations, self.n_actions[0], self.n_actions[1], self.params)
        elif self.Model == "A3C":
            """
            Asynchronous Advantage Actor-Critic (비동기식 Actor Critic 모델)
            """
            self.params["A3C"] = dict()
            self.params["A3C"]["MAX_EP"] = 3000
            self.params["A3C"]["GAMMA"] = 0.99
            self.params["A3C"]["UPDATE_GLOBAL_ITER"] = 5
            self.params["A3C"]["lr"] = 1e-4

            self.gnet = Net(self.n_observations, self.n_actions[0], self.n_actions[1])
            self.gnet.share_memory()
            opt = SharedAdam(self.gnet.parameters(), lr=self.params["A3C"]["lr"],
                            betas=(0.92, 0.999))
            self.global_ep, self.global_ep_r, self.res_queue = mp.Value('i', 0), mp.Value('d', 0.), mp.Queue()
            
            self.a3c_workers = [Worker( self.env, self.gnet, opt,self.global_ep, self.global_ep_r, self.res_queue, self.params,self.n_observations, self.n_actions[0], self.n_actions[1], i)
                                for i in range(mp.cpu_count())]
        

    def save_model(self, dqn_policy_net, episode_num):
        if not os.path.exists('output'):
            os.makedirs('output')

        save_path = f'output/{self.Model}_model_episode_{episode_num}.pth'
        if self.Model == "DQN":
            torch.save(dqn_policy_net.state_dict(), save_path)
            print(f"Model saved to episode {episode_num} to {save_path}")


    def plot_durations(self,final_x_history, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        reward_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        final_x_history = torch.tensor(final_x_history,dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        # plt.plot(durations_t.numpy())
        plt.plot(reward_t.numpy())
        # plt.plot(final_x_history.numpy())
        plt.pause(0.001)  # 도표가 업데이트되도록 잠시 멈춤
        if self.is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

            
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
        success_queue = queue.Queue(maxsize=100)
        success_rate_history = []
        final_x_history = []

        
        for i_episode in range(num_episodes):
            self.env.episode_num = i_episode + 1 
            self.steps_done = 0 
            # state : np.ndarray 
            state = self.env.reset()
            # state : torch Float32 tensor 
            if self.Model == "DQN" or "DDQN":
                state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            
            """
            Actor Critic
            """
            rewards = 0
            self.entropy = 0
            smooth = 10 
            done = False
            episode_reward = 0 
            if self.Model == "DDQN": self.ddqn.init()

            if self.Model == "DQN":
                for t in count():
                    action, self.steps_done= self.dqn.select_action(state, self.steps_done, self.env)

                    self.env.render(self.Model)
                    # if i_episode % 10 == 0 :
                    #     self.env.render()
                    observation, reward, done, is_success = self.env.step(action.squeeze(0).cpu().numpy())
                    episode_reward += reward
                    reward = torch.tensor([reward], device=self.device)
                    next_state = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
                    
                    self.dqn.memory.push(state, action, next_state, reward)
                    state = next_state
                    self.dqn.optimize_model()
                    target_net_state_dict = self.dqn.target_net.state_dict()
                    policy_net_state_dict = self.dqn.policy_net.state_dict()
                    for key in policy_net_state_dict:
                        target_net_state_dict[key] = policy_net_state_dict[key]*self.params["DQN"]["TAU"] + target_net_state_dict[key]*(1-self.params["DQN"]["TAU"])
                    self.dqn.target_net.load_state_dict(target_net_state_dict)
                    
                    
                    if done:
                        final_x_history.append(self.env.ego.x)
                        self.episode_durations.append(t + 1)
                        self.episode_rewards.append(episode_reward)
                        self.plot_durations(final_x_history)
                        if (i_episode +1) % 500 == 0:
                            self.dqn.save_model(i_episode + 1)
                        break

            elif self.Model == "DDQN":
                while not done:
                    # State ==> torch.FloatTensor   
                    # torch.Size([1, 18]) <class 'torch.Tensor'>
                    action = self.ddqn.select_action(state) # list, 2
                    next_state, reward, done, is_success = self.env.step(action)
                    # next_state => numpy.ndarray , (18,)
                    # reward => float
                    # done => bool
                    # is_success => bool
                    episode_reward += reward


                    reward = torch.tensor([reward], device=self.device)                                         # torch.Tensor, [1]
                    next_state = torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0) # torch.Tensor, [1, 18]
                    action = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)         # torch.Tensor, [1, 2]
                    ddqn_done = torch.tensor(done, dtype=torch.bool, device=self.device).unsqueeze(0)           # torch.Tensor, [1]


                    self.ddqn.memory.add((state, next_state, action, reward, ddqn_done))
                    if self.ddqn.memory.size() > 128:
                        if self.ddqn.begin_learn is False:
                            print('learn begin!')
                            self.ddqn.begin_learn = True
                        self.ddqn.learn_steps += 1 
                        if self.ddqn.learn_steps % self.ddqn.UPDATE_STEPS == 0:
                            self.ddqn.targetQNetwork.load_state_dict(self.ddqn.onlineQNetwork.state_dict())
                        batch = self.ddqn.memory.sample(self.ddqn.BATCH, False) # List

                        batch_state, batch_next_state, batch_action, batch_reward, batch_done = zip(*batch)

                        batch_state = torch.stack(batch_state).to(device)                   # torch.Tensor, [16,1,18]
                        batch_next_state = torch.stack(batch_next_state).to(device)         # torch.Tensor, [16,1,18]
                        batch_action = torch.stack(batch_action).unsqueeze(1).to(device)    # torch.Tensor, [16,1,1,2]
                        batch_reward = torch.stack(batch_reward).unsqueeze(1).to(device)    # torch.Tensor, [16,1,1]
                        batch_done = torch.stack(batch_done).unsqueeze(1).to(device)        # torch.Tensor, [16,1,1]

                        with torch.no_grad():
                            # self.ddqn.onlineQNetwork = self.ddqn.onlineQNetwork(batch_next_state)
                            # self.ddqn.targetQNetwork = self.ddqn.targetQNetwork(batch_next_state)
                            # online_max_action = torch.argmax(self.ddqn.onlineQNetwork, dim=1, keepdim=True)
                            # y = batch_reward + (1-batch_done) * self.ddqn.GAMMA * self.ddqn.targetQNetwork.gather(1, onlinse_max_action.long())
                            online_Q1, online_Q2 = self.ddqn.onlineQNetwork(batch_next_state)
                            target_Q1, target_Q2 = self.ddqn.targetQNetwork(batch_next_state)

                            online_max_action_1 = torch.argmax(online_Q1, dim=1, keepdim=True)
                            online_max_action_2 = torch.argmax(online_Q2, dim=1, keepdim=True)

                            # Bool type batch done을 float type으로 변경 
                            batch_done = batch_done.float()

                            y_1 = batch_reward + (1 - batch_done) * self.ddqn.GAMMA * target_Q1.gather(1, online_max_action_1)
                            y_2 = batch_reward + (1 - batch_done) * self.ddqn.GAMMA * target_Q2.gather(1, online_max_action_2)
                            print(f"y 1 : {y_1}, size : {y_1.size()}")
                            print(f"y 2 : {y_2}, size : {y_2.size()}")

                        
                        batch_action = batch_action.squeeze(1).squeeze(1)
                        
                        
                        online_Q1 = online_Q1.squeeze(1)
                        online_Q2 = online_Q2.squeeze(1) 


                        print(f"Online Q1 : {online_Q1}, size : {online_Q1.size()}")
                        print(f"batch action : {batch_action}, size : {batch_action.size()}")
                        


                        predicted_Q1 = online_Q1.gather(1, batch_action[:, 0].unsqueeze(1).long())
                        predicted_Q2 = online_Q2.gather(1, batch_action[:, 1].unsqueeze(1).long())
                        print("Requires grad check:")
                        print("predicted_Q1:", predicted_Q1.requires_grad)
                        print("predicted_Q2:", predicted_Q2.requires_grad)
                        print("y_1:", y_1.requires_grad)
                        print("y_2:", y_2.requires_grad)

                        loss_1 = F.mse_loss(predicted_Q1, y_1)
                        loss_2 = F.mse_loss(predicted_Q2, y_2)

                        total_loss = loss_1 + loss_2 

                    
                        # loss = F.mse_loss(self.ddqn.onlineQNetwork(batch_state).gather(1, batch_action.long()), y)


                        self.ddqn.optimizer.zero_grad()
                        total_loss.backward()
                        self.ddqn.optimizer.step()
                        self.ddqn.writer.add_scalar('loss', loss.item(), global_step=self.ddqn.learn_steps)

                        if self.ddqn.epsilon > self.ddqn.FINAL_EPSILON:
                            self.ddqn.epsilon -= (self.ddqn.INITIAL_EPSILON - self.ddqn.FINAL_EPSILON) / self.ddqn.EXPLORE
                    
                    if done:
                        final_x_history.append(self.env.ego.x)
                        self.episode_durations.append(t + 1)
                        self.episode_rewards.append(reward.item())
                        self.plot_durations(final_x_history)
                        break
                    state = next_state

                    
            elif self.Model == "PPO":
                while not done:
                    # 행동 선택 
                    action_1, action_2 = self.ppo.select_action(state)
                    # self.env.render(self.Model)
                    action = np.array([action_1, action_2])
                    # 환경에 행동을 적용 
                    next_state, reward, done, is_success = self.env.step(action)

                    episode_reward += reward 

                    # PPO 에 필요한 데이터 저장 (버퍼에 상태, 행동, 보상)
                    self.ppo.buffer.rewards.append(reward)
                    self.ppo.buffer.is_terminals.append(done)

                    # 상태 업데이트
                    state = next_state 

                    # 일정 주기로 PPO 업데이트
                    if done:
                        self.ppo.update()

                        # 에피소드 정보 기록
                        self.episode_rewards.append(episode_reward)
                        # self.plot_durations()

                        if (i_episode + 1 ) % 500 == 0:
                            self.ppo.save(i_episode+1)
                     
            # elif self.Model == "AC":
            #     for t in count():
            #         self.estimate_value(state)
            #         policy = self.actor(state).cpu().detach().numpy()
            #         action = self.select_action(state)

            #         e = - np.sum(np.mean(policy) * np.log(policy))
            #         self.entropy += e 

            #         state, reward, done = self.env.step(action.squeeze(0).cpu().numpy())
            #         rewards += reward

            #         self.actor.reward_episode.append(reward)

            #         if done:
            #             self.episode_durations.append(t + 1)
            #             self.episode_rewards.append(reward.item())
            #             self.plot_durations()
            #             # if (i_episode +1) % 500 == 0:
            #             #     self.save_model(self.policy_net, i_episode + 1)
            #             break
            #     smoothed_reward.append(rewards)
            #     if len(smoothed_reward) > smooth: 
            #         smoothed_reward = smoothed_reward[-1*smooth: -1]

            #     a_loss, c_loss = self.update_a2c()
                
            #     self.writer.add_scalar("Critic Loss", c_loss, i_episode)
            #     self.writer.add_scalar("Actor Loss", a_loss, i_episode)
            #     self.writer.add_scalar("Reward", rewards, i_episode)
            #     self.writer.add_scalar("Mean Reward", np.mean(smoothed_reward), i_episode)

            #     self.plots["Critic Loss"].append(c_loss * 100)
            #     self.plots["Actor Loss"].append(a_loss)
            #     self.plots["Reward"].append(rewards)
            #     self.plots["Mean Reward"].append(np.mean(smoothed_reward))
            # # if i_episode % 10 == 0 :
            # #         self.env.close()

            ####################  성공 확률 #################### 
            if success_queue.full():
                success_queue.get()


            success_rate = sum(list(success_queue.queue)) / 100
            success_rate_history.append(success_rate)
            self.env.success_rate = success_rate

            # self.plot_durations(final_x_history)


        """
        A3C 자체적 학습 기능 탑제
        """

        if self.Model == "A3C":
                [w.start() for w in self.a3c_workers]
                res = [] 

                while True:
                    r = self.res_queue.get()
                    if r is not None:
                        res.append(r)
                    else:
                        break
                [w.join() for w in self.a3c_workers]
                plt.plot(res)
                plt.ylabel('Moving average ep reward')
                plt.xlabel('Step')
                plt.show()        
        
        # print('Complete')
        # self.plot_durations(final_x_history=final_x_history,show_result=True)
        plt.ioff()
        plt.show()


def main():
    train_runner = Runner()
    train_runner.train()

if __name__ == "__main__":
    main()