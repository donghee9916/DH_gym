from gym import Env
from gym import spaces
from gym.spaces import Box  # Observation space 용 
import numpy as np 

class ENV(Env):
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([3,5])     # 1,2 action의 범위
        # self.action_space = [i for i in range(-2,3)]        # 예제의 Dueling DQN은 action 추론을 오직 양수로만 한다. 따라서 이렇게 설정  
        self.observation_space = Box(low=np.array([0]), high=np.arary([100]), dtype=np.int8)
        self.state = np.random.choice([-20, 0, 20, 40, 60])
        self.prev_sate = self.state

        self.episode_length = 100


    def step(self, action):
        self.state += self.action_space[action]
        self.episode_length -= 1 

        if self.state >= 20 and self.state <= 25:
            reward = 100
        else:
            reward = -100

        prev_diff = min(abs(self.prev_sate-20), abs(self.prev_sate-25))
        curr_diff = min(abs(self.state - 20), abs(self.state - 25))

        if curr_diff <= prev_diff:
            if reward != 100: reward += 50
            else: reward = 100
        if curr_diff > prev_diff:
            reward -= 50
        self.prev_state = self.state 

        if self.episode_length <= 0:
            done = True
        else:
            done = False 

        info = {}
        
        return self.get_obs(), reward, done, info 


    def reset(self):
        self.state = np.random.choice([-20, 0, 20, 40, 60])
        self.episode_length = 100
        return self.get_obs()
    
    def get_obs(self):
        return np.array([self.state], dtype=int)