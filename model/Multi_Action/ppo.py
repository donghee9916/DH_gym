import torch 
import torch.nn as nn 
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

import numpy as np 
import os 

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


"""
Multi Discrete PPO Policy
"""


class RolloutBuffer:
    def __init__(self):
        self.actions_1 = []
        self.actions_2 = []
        self.states= [] 
        self.logprobs_1 = []
        self.logprobs_2 = [] 
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
         
    def clear(self):
        del self.actions_1[:]
        del self.actions_2[:]
        del self.states[:]
        del self.logprobs_1[:]
        del self.logprobs_2[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_1_dim, action_2_dim):
        super(ActorCritic, self).__init__()

        # Actor 부분: 두 개의 독립적인 행동을 처리
        self.actor_1 = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_1_dim),  # action_1에 대한 확률 예측
                        nn.Softmax(dim=-1)
                    )

        self.actor_2 = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, action_2_dim),  # action_2에 대한 확률 예측
                        nn.Softmax(dim=-1)
                    )
            
        # Critic
        self.critic = nn.Sequential(
                        nn.Linear(state_dim, 64),
                        nn.Tanh(),
                        nn.Linear(64, 128),
                        nn.Tanh(),
                        nn.Linear(128, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def forward(self):
        raise NotImplementedError
    
    def act(self, state):
        action_probs_1 = self.actor_1(state)
        action_probs_2 = self.actor_2(state)

        dist_1 = Categorical(action_probs_1)
        dist_2 = Categorical(action_probs_2)

        # 2개의 행동을 샘플링
        action_1 = dist_1.sample()
        action_2 = dist_2.sample()

        # 행동 확률 로그 계싼 
        action_log_prob_1 = dist_1.log_prob(action_1)
        action_log_prob_2 = dist_2.log_prob(action_2)

        state_val = self.critic(state)



        return action_1.detach(), action_log_prob_1.detach(), action_2.detach(), action_log_prob_2.detach(), state_val.detach()
    
    def evaluate(self, state, action_1, action_2):
        action_probs_1 = self.actor_1(state)
        action_probs_2 = self.actor_2(state)

        dist_1 = Categorical(action_probs_1)
        dist_2 = Categorical(action_probs_2)

        action_logprobs_1 = dist_1.log_prob(action_1)
        action_logprobs_2 = dist_2.log_prob(action_2)

        dist_entropy_1 = dist_1.entropy()
        dist_entropy_2 = dist_2.entropy()

        state_values = self.critic(state)

        return action_logprobs_1, action_logprobs_2, state_values, dist_entropy_1, dist_entropy_2

        
        
class PPO:
    def __init__(self, state_dim, action_1_dim, action_2_dim, params):
        self.params = params
        self.lr_actor = self.params["PPO"]["lr_actor"]
        self.lr_critic = self.params["PPO"]["lr_critic"]
        self.gamma = self.params["PPO"]["gamma"]
        self.K_epochs = self.params["PPO"]["K_epochs"]
        self.eps_clip = self.params["PPO"]["eps_clip"]
        if self.params["PPO"]["random_seed"] != 0:
            torch.manual_seed(self.params["PPO"]["random_seed"])
            np.random.seed(self.params["PPO"]["random_seed"])
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_1_dim, action_2_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params' : self.policy.actor_1.parameters(), 'lr': self.lr_actor},
            {'params' : self.policy.actor_2.parameters(), 'lr': self.lr_actor},
            {'params' : self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_1_dim, action_2_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state): 
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            
            action_1, aciton_logprob_1, action_2, action_logprob_2, state_val = self.policy_old.act(state)
        # print(f"state size: {state.size()}, type: {type(state)}")
        self.buffer.states.append(state)
        self.buffer.actions_1.append(action_1)
        self.buffer.actions_2.append(action_2)
        self.buffer.logprobs_1.append(aciton_logprob_1)
        self.buffer.logprobs_2.append(action_logprob_2)
        self.buffer.state_values.append(state_val)

        return action_1.item(), action_2.item()
    
    def update(self):
        # Calculate Monte-Carlo Estimated Return
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0 
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0,discounted_reward)
        
        # Normalizing the reward 
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
         
        # Convert list to tensor
        # old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_states = torch.squeeze(torch.stack([torch.tensor(state, dtype=torch.float32) if isinstance(state, np.ndarray) else state for state in self.buffer.states], dim=0)).detach().to(device)
        
        # Convert actions list to tensors
        old_actions_1 = torch.tensor(self.buffer.actions_1, dtype=torch.long).to(device)  # Ensure it's LongTensor for Categorical distribution
        old_actions_2 = torch.tensor(self.buffer.actions_2, dtype=torch.long).to(device)  # Same for second action

        old_logprobs_1 = torch.tensor(self.buffer.logprobs_1, dtype=torch.float32).to(device)
        old_logprobs_2 = torch.tensor(self.buffer.logprobs_2, dtype=torch.float32).to(device)
        old_state_values = torch.tensor(self.buffer.state_values, dtype=torch.float32).to(device)
        # old_actions_1 = torch.squeeze(torch.stack(self.buffer.actions_1, dim=0)).detach().to(device)
        # old_logprobs_1 = torch.squeeze(torch.stack(self.buffer.logprobs_1, dim=0)).detach().to(device)
        
        # old_actions_2 = torch.squeeze(torch.stack(self.buffer.actions_2, dim=0)).detach().to(device)
        # old_logprobs_2 = torch.squeeze(torch.stack(self.buffer.logprobs_2, dim=0)).detach().to(device)
        # old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        



       

        # Calculate Advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            
            # Evaluate old actions and values 
            logprobs_1, logprobs_2, state_values, dist_entropy_1, dist_entropy_2 = self.policy.evaluate(old_states, old_actions_1, old_actions_2)
            
            # Match state_values tensor dimensions with reward tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta_old)
            ratios_1 = torch.exp(logprobs_1 - old_logprobs_1.detach())
            ratios_2 = torch.exp(logprobs_2 - old_logprobs_2.detach())

            # Finding Surrogate Loss 
            surr1_1 = ratios_1 * advantages
            surr1_2 = torch.clamp(ratios_1, 1-self.eps_clip, 1+self.eps_clip) * advantages

            surr2_1 = ratios_2 * advantages
            surr2_2 = torch.clamp(ratios_2, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # Finding Surrogate Loss
            loss_1 = -torch.min(surr1_1, surr1_2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy_1
            loss_2 = -torch.min(surr2_1, surr2_2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy_2


            # final loss of clipped objective PPO
            loss = loss_1 + loss_2 

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # Clear Buffer          
        self.buffer.clear()   

    def save(self, episode_num):
        if not os.path.exists('output'):
            os.makedirs('output')

        save_path = f'output/ppo_model_episode_{episode_num}.pth'
        torch.save(self.policy_old.state_dict(), save_path)
        print(f"Model saved at episode {episode_num} to {save_path}")