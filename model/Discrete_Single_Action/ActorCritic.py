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

device =  torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module): 
  def __init__(self, in_size, out_size): 
    super(Actor, self).__init__()
    self.linear1 = nn.Linear(in_size, 128)
    self.linear2 = nn.Linear(128, out_size)
    self.dropout = nn.Dropout(0.7)
    self.softmax = nn.Softmax(dim= 1)

    self.policy_history = Variable(torch.Tensor()).to(device)
    self.reward_episode = []

    self.reward_history = []
    self.loss_history = []
  
  def forward(self, x): 
    #convert numpy state to tensor
    x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
    x = F.relu(self.linear1(x))
    x = self.dropout(x)
    x = self.softmax(self.linear2(x))
    return x

#critic network
class Critic(nn.Module): 
  def __init__(self, in_size): 
    super(Critic, self).__init__()
    self.linear1 = nn.Linear(in_size, 128)
    self.linear2 = nn.Linear(128, 1)
    self.dropout = nn.Dropout(0.7)

    self.value_episode = []
    self.value_history = Variable(torch.Tensor()).to(device)
    
  def forward(self, x): 
    x = Variable(torch.from_numpy(x).float().unsqueeze(0)).to(device)
    x = F.relu(self.linear1(x))
    x = self.linear2(x)
    return x 

#combined module (mostly for loading / storing)
class ActorCritic(nn.Module): 
  def __init__(self, actor, critic): 
    super(ActorCritic, self).__init__()
    self.actor = actor
    self.critic = critic
  
  def forward(self, x):
    value = self.critic(x)
    policy = self.actor(x)

    return value, policy