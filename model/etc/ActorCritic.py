import torch 
import torch.nn as nn 
import torch.nn.functional as F 

class Actor(nn.Module):
    def __init__(self, n_observation, action_shapes):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(n_observation, 128)
        self.layer2 = nn.Linear(128, 128)

        self.layer_action_1 = nn.Linear(128, 3)
        self.layer_action_2 = nn.Linear(128, 5)

    def forward(self,x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        
        action_1 = self.layer_action_1(x)
        action_2 = self.layer_action_2(x)
        action_values = torch.cat([action_1, action_2], dim=1)

        return action_values

class Critic(nn.Module):
    def __init__(self, n_observation):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(n_observation, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer_value = 