# model/actor_critic.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    def __init__(self, n_observations, action_shapes):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        
        # 각 액션 공간에 대해 별도의 출력 레이어
        self.layer_action_1 = nn.Linear(128, 3)  # 첫 번째 액션 공간
        self.layer_action_2 = nn.Linear(128, 5)  # 두 번째 액션 공간

        self.softmax = nn.Softmax(dim=1)

        self.policy_history = Variable(torch.Tensor()).to(device)
        self.reward_episode = []

        self.reward_history= [] 
        self.loss_history = []

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
    
        
        action_1_probs = F.softmax(self.layer_action_1(x), dim=-1)  # 첫 번째 액션의 확률
        action_2_probs = F.softmax(self.layer_action_2(x), dim=-1)  # 두 번째 액션의 확률
        
        action_values = torch.cat([action_1_probs, action_2_probs], dim=1)
        return action_values



class Critic(nn.Module):
    def __init__(self, n_observations):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 1)  # 상태 가치 출력 (Scalar)

        self.value_episode = []
        self.value_history = Variable(torch.Tensor()).to(device)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        return self.layer3(x)  # Single scalar output: value of the state

class ActorCritic(nn.Module):
    def __init__(self, n_observations, action_shapes):
        super(ActorCritic, self).__init__()
        self.actor = Actor(n_observations, action_shapes)
        self.critic = Critic(n_observations)

    def forward(self, x):
        action_probs = self.actor(x)
        value = self.critic(x)
        return action_probs, value
