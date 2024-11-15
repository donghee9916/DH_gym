import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torch.multiprocessing as mp
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
from .utils import v_wrap, set_init, push_and_pull, record
from Environment.env import Env

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
Asynchronous Advantage Actor-Critic (비동기식 Actor Critic 모델 --> AC의 대표적 모델)

Global Network와 Local network 간의 데이터를 주고 받는 방식으로 학습을 병렬적으로 수행 

Multi Discrete A3C 

"""
env= Env(18)

class Net(nn.Module):
    def __init__(self, state_dim, action_1_dim, action_2_dim):
        """
        Local Network로 Actor Critic 모델을 정의한 것 
        """
        super(Net, self).__init__()
        
        self.pi1 = nn.Linear(state_dim, 128)
        
        self.pi2_1 = nn.Linear(128, action_1_dim)
        self.pi2_2 = nn.Linear(128, action_2_dim)

        self.v1 = nn.Linear(state_dim,128)
        self.v2 = nn.Linear(128,1)
        set_init([self.pi1, self.pi2_1, self.pi2_2, self.v1, self.v2])
        self.distribution = torch.distributions.Categorical

    def forward(self,x):
        pi1 = torch.tanh(self.pi1(x))
        
        logits_1 = self.pi2_1(pi1)
        logits_2 = self.pi2_2(pi1)

        v1 = torch.tanh(self.v1(x))
        values = self.v2(v1)
        return logits_1, logits_2, values
    
    def choose_action(self, s):
        self.eval()
        logits_1, logits_2, _ = self.forward(s)
        prob_1 = F.softmax(logits_1, dim=1).data
        prob_2 = F.softmax(logits_2, dim=1).data
        m_1 = self.distribution(prob_1)
        m_2 = self.distribution(prob_2)
        return m_1.sample().numpy()[0], m_2.sample().numpy()[0]
        
    def loss_func(self, s, a, v_t):
        self.train()
        logits_1, logits_2, values = self.forward(s) 
        td = v_t - values
        c_loss = td.pow(2)

        prob_1 = F.softmax(logits_1, dim=1).data
        prob_2 = F.softmax(logits_2, dim=1).data
        m_1 = self.distribution(prob_1)
        m_2 = self.distribution(prob_2)
        exp_v_1 = m_1.log_prob(a) * td.detach().squeeze()
        exp_v_2 = m_2.log_prob(a) * td.detach().squeeze()
        a_loss_1 = -exp_v_1
        a_loss_2 = -exp_v_2

        total_loss = (c_loss + a_loss_1).mean() + (c_loss + a_loss_2).mean()
        return total_loss



class Worker(mp.Process):
    def __init__(self, env, gnet, opt,global_ep, global_ep_r, res_queue, params,state_dim, action_1_dim, action_2_dim, name):
        super(Worker, self).__init__()
        self.params = params
        self.name = 'w%02i' % name
        self.MAX_EP = self.params["A3C"]["MAX_EP"]
        self.GAMMA = self.params["A3C"]["GAMMA"]
        self.UPDATE_GLOBAL_ITER = self.params["A3C"]["UPDATE_GLOBAL_ITER"]
        self.gnet, self.opt = gnet, opt
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.lnet = Net(state_dim, action_1_dim, action_2_dim)           # local network

    def run(self):
        total_step = 1 
        while self.g_ep.value < self.MAX_EP:
            s = env.reset()
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0.
            while True:
                a = self.lnet.choose_action(v_wrap(s[None, :]))
                s_, r, done, _ = env.step(a)
                if done: r = -1
                ep_r += r
                buffer_a.append(a)
                buffer_s.append(s)
                buffer_r.append(r)

                if total_step % self.UPDATE_GLOBAL_ITER == 0 or done:  # update global and assign to local net
                    # sync
                    push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, self.GAMMA)
                    buffer_s, buffer_a, buffer_r = [], [], []

                    if done:  # done and print information
                        record(self.g_ep, self.g_ep_r, ep_r, self.res_queue, self.name)
                        break
                s = s_
                total_step += 1
        self.res_queue.put(None)