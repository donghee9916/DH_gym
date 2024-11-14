import sys
import os 
import csv
sys.path.append(__file__)
import numpy as np 
from datetime import datetime
import matplotlib.pyplot as plt
from .cm_utils import CarMakerWithRL 
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import torch.optim as optim

import rclpy
from rclpy.node import Node
from std_msgs.msg import Int32, Float64MultiArray, Float64

from .PPO_AMLAB import PPO, ActorCritic, RolloutBuffer
from .DQN_AMLAB import DQNAgent
from .Actor_Critic import ActorCritic, Actor, Critic, Actor_Critic_Runner
from .dl_class  import *

from custom_msgs.msg import Rlstate, Episodeflag, MoraiMSC
from std_msgs.msg import Int32

import time 

# ROS2 노드 정의
class LaneChangeController(Node):
    def __init__(self):
        super().__init__('lane_change_controller')


        ################################## set device ##################################
        print("============================================================================================")
        self.device = torch.device('cpu')
        if(torch.cuda.is_available()): 
            self.device = torch.device('cuda:0') 
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(self.device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")


        ################################## set Learning Name ################################
        self.RL_NAME = "LC_DL_1102"
        self.EPISODE_DIR_NAME = 'src/dh_rl_python/models'
        self.LOG_DIR_NAME = 'src/dh_rl_python/logs'
        os.makedirs(self.LOG_DIR_NAME, exist_ok=True)
        os.makedirs(self.EPISODE_DIR_NAME, exist_ok=True)


        # 텍스트 파일 로그 설정
        self.text_log_path = os.path.join(self.LOG_DIR_NAME, self.RL_NAME + '_training_log.txt')

        # CSV 파일 로그 설정
        self.csv_log_path = os.path.join(self.LOG_DIR_NAME, self.RL_NAME + '_training_log.csv')
        self.csv_file = open(self.csv_log_path, mode = 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Episode', 'Reward', 'Avg Reward', 'Timestamp'])

        # Episode reward 초기화 
        self.current_ep_reward = 0
        self.episode_num = 1

        self.prev_state = RL_State(self.device)
        self.current_state = RL_State(self.device)


        ################################### Learning Setting ##################################### 
        self.max_training_episodes = 3000
        self.print_freq = 10
        self.save_model_freq = 10
        self.update_freq = 5

        ################################## Agent Setting ########################################
        self.Mode = "A2C"   # PPO, DQN, A2C


        """
        PPO Setting
        """
        state_dim = 9
        action_dim = 9
        has_continuous_action_space = False
        action_std = 0.0

        K_epochs = 80
        eps_clip = 0.20
        gamma = 0.99

        lr_actor = 0.0003 
        lr_critic = 0.001

        random_seed = 0 
        action_std = 0.6                    # starting std for action distribution (Multivariate Normal
        action_decay_rate = 0.05            # linearly decay action_std (action_std = action_std - action_std_decay_rate)
        min_action_std = 0.1                # minimum action_std (stop decay after action_std <= min_action_std)
        action_std_decay_freq = 100000      # action_std decay frequency (in num timesteps)    

        # self.ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, has_continuous_action_space, action_std)

        """
        DQN Setting
        """
        self.state_dim = 34     # 
        self.action_dim = 7     # LL, L, SL, N, SR, R, LR 총 7개
        self.state = torch.FloatTensor(np.zeros(self.state_dim)).to(self.device)
        lr = 0.001
        gamma = 0.99
        epsilon = 0.1
        target_update = 10
        memory_capacity = 20000
        batch_size = 64
        
        self.dqn_agent = DQNAgent(self.state_dim, self.action_dim, lr, gamma, epsilon, target_update, memory_capacity, batch_size)


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

        self.actor = Actor(self.state_dim, self.action_dim, hidden_units, dropout_rate)
        self.critic = Critic(self.state_dim, hidden_units, dropout_rate)
        self.ac = ActorCritic(self.actor, self.critic)

        a_optimizer = optim.Adam(self.actor.parameters(), lr = lr)
        c_optimizer = optim.Adam(self.critic.parameters(), lr = lr )

        self.ac_runner = Actor_Critic_Runner(self.actor, self.critic, a_optimizer, c_optimizer, gamma=gamma,
                                    entropy_coeff=entropy_coeff, value_loss_coeff=value_loss_coeff, 
                                    actor_loss_coeff=actor_loss_coeff, gradients_clipping=gradients_clipping, 
                                    logs="a2c_decision/%s" % time.time())

    

        ################################## SUB/PUB Setting ########################################
        """
        SUBSCRIBER
        """
        self.sub_state = self.create_subscription(Rlstate, '/rl_state', self.rl_state_callback, 10)
        self.sub_episode_flag = self.create_subscription(Episodeflag, '/rl_episode_flag', self.rl_episode_flag_callback,10)
        # self.sub_lap_num = self.create_subscription(Int32, 'decision/lap_num', self.lap_num_callback,10)
        self.sub_morai_msc = self.create_subscription(MoraiMSC, '/moari/msc', self.morai_msc_callback, 10)
        
        # self.collision_subscriber = self.create_subscription(Int32, '/morai/collision', self.collision_callback, 10)
        # self.initialize_flag_subscriber = self.create_subscription(Int32, '/morai/initialize_flag', self.initialize_flag_callback,10)
        """
        PUBLISHER
        """
        self.pub_rl_action = self.create_publisher(Int32, '/rl_action', 1)
        self.pub_v2x_can = self.create_publisher(Float64MultiArray, "/v2x/racing_flag_can", 1)
        self.pub_velocity = self.create_publisher(Float64, '/decision/ref_velocity',1)
        """
        TIMER Callback
        """
        self.timer_ = self.create_timer(0.01, self.timer_callback)

        """
        Callback DATA
        """
        ################################### ADDITIONAL INFOS ########################################
        self.start_time = datetime.now().replace(microsecond=0)
        self.time = datetime.now().replace(microsecond=0)


        self.print_running_reward = 0
        self.print_running_episodes = 0

        self.current_lap_num = 0
        self.end_flag = False
        self.initializing_complete_flag = True
        self.init_flag = True
        self.collision_flag = False


        self.TAG = "SCENE_4_Multi_Vehicle"
        self.model_dir = os.path.join("src/dh_rl_python/dh_rl_python/output", self.TAG)
        os.makedirs(self.model_dir, exist_ok=True)

        # 최상의 모델과 최신 모델 경로 설정
        self.best_model_path = os.path.join(self.model_dir, "best.pth")
        self.latest_model_path = os.path.join(self.model_dir, "latest.pth")
        self.saved_models = []

        self.best_reward = 0.0

        self.lap_time = time.time()

        self.reward_history = []
        self.prev_action = None
        self.action = None


        # # Load Best pth 
        # if os.path.exists(self.best_model_path):
        #     print("Loading the best model: best.pth")
        #     self.load_model(self.best_model_path, self.best_model_path)  # Load both policy and target nets
        # else:
        #     print("No best.pth found. Starting with a new model.")



    def save_model(self, episode_num):
        today = datetime.now().strftime('%y%m%d')
        directory = os.path.join(self.model_dir, today)
        os.makedirs(directory, exist_ok=True)

        # 파일 경로 설정
        policy_net_path = os.path.join(directory, f'dqn_policy_net_{episode_num}.pth')
        target_net_path = os.path.join(directory, f'dqn_target_net_{episode_num}.pth')

        # 모델 저장
        torch.save(self.dqn_agent.policy_net.state_dict(), policy_net_path)
        torch.save(self.dqn_agent.target_net.state_dict(), target_net_path)
        print(f"모델 가중치 저장됨 : 에피소드 {episode_num}")

        # 최신 모델로 설정
        torch.save(self.dqn_agent.policy_net.state_dict(), self.latest_model_path)
        print("최신 모델로 저장됨: latest.pth")

        # 최고 보상 모델로 설정
        if self.current_ep_reward > self.best_reward:
            self.best_reward = self.current_ep_reward
            torch.save(self.dqn_agent.policy_net.state_dict(), self.best_model_path)
            print("최고 모델로 갱신됨: best.pth")

        # 최근 5개 모델 관리
        self.saved_models.append(policy_net_path)
        if len(self.saved_models) > 5:
            oldest_model_path = self.saved_models.pop(0)
            os.remove(oldest_model_path)
            print(f"가장 오래된 모델 삭제됨: {oldest_model_path}")

    def morai_msc_callback(self, msg):
        self.end_flag = msg.end_flag
        self.initializing_complete_flag = msg.initializing_complete_flag
        self.collision_flag = msg.collision_flag

    def rl_state_callback(self, msg):

        
            ################################## STATE SUBSCRIBTION #####################################
            # 추가적으로 필요한 정보
            """
            1. 이전 action 정보
            2. 현재 action 정보
            3. 현재 제한 속도 정보
            4. spd up / down을 위한 정보 
            """
            self.state = np.zeros(self.state_dim)
            # 각각의 값들을 배열의 특정 인덱스에 직접 할당
            self.state[0] = msg.ego_velocity
            self.state[1] = msg.ego_lane_order
            self.state[2] = msg.gpp_lane_order
            self.state[3] = msg.speed_limit
            self.state[4] = msg.ego_prev_action
            self.state[5] = msg.maximum_lane_num

            # front_ttc_array, rear_ttc_array, front_distance_array, rear_distance_array의 값을 하나씩 할당
            self.state[6:6+len(msg.front_ttc_array)] = msg.front_ttc_array
            self.state[6+len(msg.front_ttc_array):6+2*len(msg.front_ttc_array)] = msg.rear_ttc_array
            self.state[6+2*len(msg.front_ttc_array):6+3*len(msg.front_ttc_array)] = msg.front_distance_array
            self.state[6+3*len(msg.front_ttc_array):6+4*len(msg.front_ttc_array)] = msg.rear_distance_array

            self.state_dim = self.state.shape[0]
            self.state = torch.FloatTensor([self.state]).to(self.device)

            # action = self.ppo_agent.select_action(self.state)
            self.reward = 0
            self.reward = torch.FloatTensor([self.reward]).to(self.device)

            ################################## DQN Based Deep Learning ####################################
            
    def rl_episode_flag_callback(self, msg):
        pass
        # self.init_flag = msg.init_flag
        # self.end_flag = msg.end_flag
        # print(f"flag subscribed init: {self.init_flag}, end: {self.end_flag}")

    def state_reset(self):
        self.state = np.zeros(self.state_dim)
        self.state = torch.FloatTensor([self.state]).to(self.device)
        self.time = datetime.now().replace(microsecond=0)
    def load_model(self, policy_net_path, target_net_path):
        self.dqn_agent.policy_net.load_state_dict(torch.load(policy_net_path))
        self.dqn_agent.target_net.load_state_dict(torch.load(target_net_path)) 
    def __del__(self):
        self.csv_file.close()

    def timer_callback(self):

        ref_vel_msg = Float64()


        if self.episode_num < self.max_training_episodes:

            if self.initializing_complete_flag :
                
                # 초기에 차량이 정지해 있을 수 있도록 하는 요소 
                ref_vel_msg.data = 30.0/3.6
                v2x_msg = Float64MultiArray()
                v2x_msg.data = [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                self.pub_v2x_can.publish(v2x_msg)


                if self.init_flag:
                    self.state_reset()
                    print(f"이전 Episode reward : {self.current_ep_reward}")
                    self.current_ep_reward = 0
                    print("Initializing Completed")
                    self.init_flag = False


                if self.prev_state.mode == 1:
                    self.prev_state.update_current_step(self.state, self.action)
                else:
                    self.prev_state.update_next_step(self.state, self.reward)
                    
                    if self.Mode == "DQN":
                        self.dqn_agent.memory.push(
                                torch.tensor([self.prev_state.state], dtype=torch.float32, device=self.device),
                                torch.tensor([[self.prev_state.action]], device=self.device),
                                torch.tensor([self.prev_state.next_state], dtype=torch.float32, device=self.device),
                                torch.tensor([self.prev_state.reward], dtype=torch.float32, device=self.device)
                            )
                    elif self.Mode == "A2C":

                        self.ac_runner.actor.reward_episode.append(self.reward)
                        self.prev_state.update_current_step(self.state, self.action)
                        actor_loss, critic_loss = self.ac_runner.update_a2c()
                        if self.episode_num % 10 == 0: 
                            print("\tEpisode {} \t Final Reward {:.2f}".format(self.episode_num, self.current_ep_reward))




                ## SELECT ACTION ##
                if self.Mode == "DQN":
                    self.action = self.dqn_agent.select_action(self.state).unsqueeze(0)
                    self.reward = self.dqn_agent.calculate_reward(self.state, self.action, self.lap_time, self.collision_flag, self.prev_action)    
                    
                    self.prev_action = self.action

                elif self.Mode == "A2C":
                    # 1. Estimate Value 
                
                    self.ac_runner.estimate_value(self.state)
                    policy = self.ac_runner.actor(self.state).cpu().detach().numpy()
                    self.action = self.ac_runner.select_action(self.state)
                    self.prev_action = self.action
                    e = -np.sum(np.mean(policy) * np.log(policy))
                    self.ac_runner.entropy += e 

                    self.reward = self.ac_runner.calculate_reward(self.state, self.action, self.lap_time, self.collision_flag, self.prev_action)

                    self.current_ep_reward += self.reward 

                    if self.episode_num % 10 == 0 :
                        print(f"Episode {self.episode_num}, Actor Loss : {actor_loss.item()}, Critic Loss: {critic_loss.item()}")

                    



                
                # 강화학습 STATE
                rl_action_msg = Int32()
                rl_action_msg.data = self.action.item()
                self.pub_rl_action.publish(rl_action_msg)
                
                if self.Mode == "DQN":
                    self.dqn_agent.optimize_model()
                self.current_ep_reward += self.reward
                # print(f"Current Reward : {self.current_ep_reward}")

                if self.collision_flag:
                    print("Collision!!!"*3)

                # 충돌이 일어나거나 종료 flag가 들어온 경우 
                if (self.end_flag or self.collision_flag):
                    print("\n##################")
                    print("Episode End")
                    self.reward_history.append(self.current_ep_reward.cpu().numpy())
                    self.init_flag = True
                    self.initializing_complete_flag = False
                    self.end_flag = False
                    self.collision = False

                    # self.episode_num += 1
                    # if self.episode_num % self.dqn_agent.target_update == 0:
                    #     self.dqn_agent.target_net.load_state_dict(self.dqn_agent.policy_net.state_dict())
                    # print(f"Episode: {self.episode_num}, Reward: {self.current_ep_reward}\n")

                    # # 에피소드 리워드 CSV 파일에 기록
                    # self.csv_writer.writerow([self.episode_num, self.current_ep_reward])

                    # # 주기적으로 모델 저장 
                    # if self.episode_num % self.save_model_freq == 0:
                    #     self.save_model(self.episode_num)


                    if self.episode_num % 10 == 0:
                        self.plot_rewards()

            else :
                v2x_msg = Float64MultiArray()
                v2x_msg.data = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0]
                self.pub_v2x_can.publish(v2x_msg)
                ref_vel_msg.data = 0.0
                print("시나리오 초기화 대기중")
        self.pub_velocity.publish(ref_vel_msg)

    def plot_rewards(self):
        # 그래프 그리기
        plt.plot(range(1, len(self.reward_history) + 1), np.array(self.reward_history))
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Episode Reward History')

        # 그래프 저장
        graph_filename = os.path.join(self.model_dir, "reward_history.png")
        plt.savefig(graph_filename)
        plt.close()
def main(args=None):

    rclpy.init(args=args)
    lane_change_controller = LaneChangeController()
    rclpy.spin(lane_change_controller)
    lane_change_controller.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()