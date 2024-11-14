import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Actor(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128,n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        distribution = Categorical(F.softmax(x, dim=-1))
        return distribution

class Critic(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)


class Runner():
    def __init__(self, n_observations, n_actions):
        self.actor = Actor(n_observations, n_actions)
        self.critic = Critic(n_observations)


        self.optimizerA = optim.Adam(self.actor.parameters())
        self.optimizerC = optim.Adam(self.critic.parameters())
    
    def train(self,env, num_episode):
        """
        Training One Episode 
        """
        for episode in range(num_episode):
            state = env.reset()
            log_probs = []
            values = [] 
            rewards = [] 
            masks = [] 
            entropy = 0 
            env.reset()

            while True:
                state = torch.FloatTensor(state).to(device)
                dist, value = self.actor(state), self.critic(state)
                action = dist.sample()
                next_state, reward, done =  env.step(action.cpu().numpy())

                log_prob = dist.log_prob(action).unsqueeze(0)
                entropy += dist.entropy().mean()

                log_probs.append(log_prob)
                values.append(value)
                rewards.append(torch.tensor([reward], dtype=torch.float, device=device))
                masks.append(torch.tensor([1-done], dtype=torch.float, device=device))

                state = next_state

                if done:
                    break
            
        next_state  = torch.FloatTensor(next_state).to(device)
        next_value = self.critic(next_state)
        returns = compute_returns(next_value, rewards, masks)

        log_probs = torch.cat(log_probs)
        returns = torch.cat(returns).detach()
        values = torch.cat(values)

        advantage = returns - values

        
                

        


def compute_returns(next_value, rewards, masks, gamma=0.99):
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def trainIters(actor, critic, n_iters):
    optimizerA = optim.Adam(actor.parameters())
    optimizerC = optim.Adam(critic.parameters())


    
