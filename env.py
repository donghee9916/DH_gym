import numpy as np
import matplotlib.pyplot as plt
from vehicle_model import BicycleModel
from purepursuit import PurePursuitController
from utils import *  # 충돌 체크 함수 및 Bezier 관련 함수들
from env_classes import *
import random
import time

import gym
from gym import spaces

import torch
class Env(gym.Env):
    def __init__(self, state_dim):
        super(Env, self).__init__()
        # self.action_space = spaces.Discrete(3) # 단일 action
        self.action_space = spaces.MultiDiscrete([3,5]) # 2가지 action
        
        # 연속적인 행동 공간을 하고 싶다면 ?
        # self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)



        self.num_lanes = 3
        self.lane_width = 3 
        self.state_dim = state_dim
        self.state = np.zeros(self.state_dim)


        self.map = Map(self.num_lanes, self.lane_width)


        # 차량 속도 (kph -> m/s로 변환)
        speed_kph = 0.0  # 차량 속도 (kph)
        speed_mps = speed_kph * 1000 / 3600  # m/s

        # 시간 단계
        self.time_step = 0.02  # 20 ms


        # 차량 및 장애물 객체 생성
        self.ego = BicycleModel(self.map.lanes[self.num_lanes//2+1][0][0], self.map.lanes[self.num_lanes//2+1][1][0],
                                heading=0.0, wheelbase=1.825, dt = self.time_step)
        self.controller = PurePursuitController(lookahead_distance= 15.0, wheelbase=1.825, )
        
        
        self.lane_order = self.num_lanes//2 + 1 
        self.collision_flag = False
        self.ep_end_flag = False
        

        # GPP 경로 시각화
        self.fig, self.ax = plt.subplots(figsize=(10, 6))

        # self.obs_1 = Object(self.map.x, self.map.lanes[num_lanes//2 + 1][1], is_moving=False)
        # self.obs = self.generate_random_obstacles(7)
        self.obs = self.generate_static_obstacles()
        self.lane_obj_dict = dict()
        self.done = False
        self.is_lane_changing = False
        self.prev_action = 0

        self.lpp_x = []
        self.lpp_y = []



        self.visualize(real_time=False)

    def reset(self):
        self.ego = BicycleModel(self.map.lanes[self.num_lanes//2+1][0][0], self.map.lanes[self.num_lanes//2+1][1][0],
                                heading=0.0, wheelbase=1.825, dt = self.time_step)
        self.controller = PurePursuitController(lookahead_distance= 15.0, wheelbase=1.825, )
        self.lane_order = self.num_lanes//2 + 1 
        self.done = False
        self.is_lane_changing = False
        self.collision_flag = False
        self.ep_end_flag = True
        self.prev_action = 0
        self.state = np.zeros(self.state_dim)
        obj_dict, dist_dict, ttc_dict = self.find_obs(self.obs,self.ego)
        self.state[0] = self.ego.vx
        self.state[1] = self.get_lane_for_object(self.ego.y)
        self.state[2] = self.lane_order
        self.state[3] = 100/3.6     # Spd Limit
        self.state[4] = self.prev_action
        self.state[5] = self.num_lanes  # 차선의 수 

        self.state[6] = ttc_dict["LF"]
        self.state[7] = ttc_dict["CF"]
        self.state[8] = ttc_dict["RF"]
        self.state[9] = ttc_dict["LR"]
        self.state[10] = ttc_dict["CR"]
        self.state[11] = ttc_dict["RR"]

        self.state[6] = dist_dict["LF"]
        self.state[7] = dist_dict["CF"]
        self.state[8] = dist_dict["RF"]
        self.state[9] = dist_dict["LR"]
        self.state[10] = dist_dict["CR"]
        self.state[11] = dist_dict["RR"]

        self.state = np.array(self.state)
        return self.state
        

    def generate_static_obstacles(self, num_obstacles = 20):
        obstacles = []
        
        lane_orders = [1,2,3]
        id = 0 
        lane_index = 0
        obj_x = 0.0
        while obj_x <= 180:
            id +=1 
            lane_order = lane_orders[lane_index]
            lane_y = self.map.lanes[lane_order][1][0]
            obj_x += 20.0 
            vx = 0.0
            vy = 0.0 
            is_moving = False 
            obstacle = Object(id = id, x = obj_x, y = lane_y, vx=vx,vy=vy, lane_order=lane_order,time_step=self.time_step,is_moving=is_moving)
            
            obstacles.append(obstacle)

            
            lane_index += 1 
            if lane_index >=3:
                lane_index -=3 
            
            id += 1 
            lane_order = lane_orders[lane_index]
            lane_y = self.map.lanes[lane_order][1][0]
            vx = 0.0
            vy = 0.0 
            is_moving = False 
            obstacle = Object(id = id, x = obj_x, y = lane_y, vx=vx,vy=vy, lane_order=lane_order,time_step=self.time_step,is_moving=is_moving)

            lane_index += 1 
            if lane_index >=3:
                lane_index -=3 
        
            obstacles.append(obstacle)

        return obstacles
        

        


    def generate_random_obstacles(self, num_obstacles):
        obstacles = []
        for _ in range(num_obstacles):
            id = _ + 1 
            lane_order = random.choice(range(1, self.num_lanes + 1))
            # 해당 차선의 y 좌표를 선택
            lane_y = self.map.lanes[lane_order][1][0]  # 해당 차선의 고정된 y값
            # 해당 차선에서 랜덤한 x 좌표 선택
            lane_x_min, lane_x_max = min(self.map.lanes[lane_order][0]), 200.0
            obj_x = random.uniform(lane_x_min, lane_x_max)

            vx = 0.0
            vy = 0.0 

            # 장애물의 이동 여부
            # is_moving = random.choice([True, False])
            is_moving = False
            # 장애물 객체 생성
            obstacle = Object(id = id, x=obj_x, y=lane_y, vx=vx, vy=vy,lane_order=lane_order, time_step=self.time_step,is_moving=is_moving)
            obstacles.append(obstacle)
            
        return obstacles
    
    def get_lane_for_object(self, obj_y):
        lane_centers = np.linspace(-(self.num_lanes // 2) * self.lane_width, (self.num_lanes // 2) * self.lane_width, self.num_lanes)
        
        for i, lane_center in enumerate(lane_centers):
            if obj_y > lane_center - self.lane_width/2 and obj_y <= lane_center + self.lane_width / 2:
                return i+1
        
        return -1 
    
    def find_obs(self, obs, ego):
        ego_obj_dict = {
            "LF": [], "CF": [], "RF": [],
            "LR": [], "CR": [], "RR": []
        }

        ego_dist_dict = {
            "LF": float(999.0), "CF": float(999.0), "RF": float(999.0),
            "LR": float(999.0), "CR": float(999.0), "RR": float(999.0)
        }

        ego_ttc_dict = {
            "LF": float(999.0), "CF": float(999.0), "RF": float(999.0),
            "LR": float(999.0), "CR": float(999.0), "RR": float(999.0)
        }

        lane_offset = {self.lane_order-1: ["LF", "LR"], self.lane_order: ["CF", "CR"], self.lane_order+1: ["RF", "RR"]}

        for obj in obs:
            lane_id = self.get_lane_for_object(obj.y)
            if lane_id in lane_offset:
                # Determine whether it's in front or behind the ego vehicle
                if obj.x >= ego.x:
                    lane = lane_offset[lane_id][0]  # Front (LF, CF, RF)
                else:
                    lane = lane_offset[lane_id][1]  # Rear (LR, CR, RR)
                
                # Append the object to the appropriate list
                ego_obj_dict[lane].append(obj)
                
                # Calculate distance and Time-to-Collision (TTC)
                dist = np.hypot(ego.x - obj.x, ego.y - obj.y)
                if dist < ego_dist_dict[lane]:
                    ego_dist_dict[lane] = dist
                    ego_ttc_dict[lane] = dist / (ego.vx - obj.vx) if (ego.vx - obj.vx) != 0 else float('inf')

        return ego_obj_dict, ego_dist_dict, ego_ttc_dict


            
    
    def visualize(self, real_time = False):


        # 실시간 시뮬레이션
        if real_time:
            plt.ion()  # Interactive mode 활성화


        self.ax.cla()
        
        ## EGO ##
        ego_rect_x = [corner[0] for corner in self.ego.rotated_corners] + [self.ego.rotated_corners[0][0]]
        ego_rect_y = [corner[1] for corner in self.ego.rotated_corners] + [self.ego.rotated_corners[0][1]]
        self.ax.plot(ego_rect_x, ego_rect_y, color='blue', label="Ego Vehicle")
        self.ax.scatter(self.ego.x, self.ego.y, color='black')  # 현재 차량 위치 표시
        self.ax.legend()
        
        ## OBJECT ##
        for obj in self.obs:
            # 장애물의 차선 구분
            lane_id = self.get_lane_for_object(obj.y)

            if lane_id == self.lane_order - 1:
                # 좌측 전방(LF) 또는 좌측 후방(LR)
                if obj.x >= self.ego.x:
                    color = 'red'  #'blue'  # 좌측 전방은 파란색
                else:
                    color = 'gray'  # 좌측 후방은 회색
            elif lane_id == self.lane_order:
                # 중앙 전방(CF) 또는 중앙 후방(CR)
                if obj.x >= self.ego.x:
                    color = 'red' # 'green'  # 중앙 전방은 초록색
                else:
                    color = 'gray' # 'lightgray'  # 중앙 후방은 연회색
            elif lane_id == self.lane_order + 1:
                # 우측 전방(RF) 또는 우측 후방(RR)
                if obj.x >= self.ego.x:
                    color = 'red'  # 우측 전방은 빨간색
                else:
                    color = 'gray' # 'darkgray'  # 우측 후방은 진회색
            else:
                color = 'black'  # 그 외의 경우 (이 경우에는 차량이 아닌 다른 장애물이라 가정)

            # 장애물 사각형 시각화
            obs_rect_x = [corner[0] for corner in obj.rotated_corners] + [obj.rotated_corners[0][0]]
            obs_rect_y = [corner[1] for corner in obj.rotated_corners] + [obj.rotated_corners[0][1]]
            self.ax.plot(obs_rect_x, obs_rect_y, color=color)  # 지정된 색으로 장애물 그리기
            self.ax.scatter(obj.x, obj.y, color=color, s=100)  # 장애물 위치 표시

        
        # 차선 표시 
        self.ax.plot(self.lpp_x, self.lpp_y, color='green', label="Local Path", linewidth=3)
    
        for lane_id, (x, y) in self.map.lanes.items():
            self.ax.plot(x, y, linestyle='--', label=f'Lane {lane_id}', color='black')
        
        # 차량 중심을 기준으로 범위 설정
        self.ax.set_xlim(self.ego.x - 30, self.ego.x + 30)  # x축은 차량 중심 ±30m
        self.ax.set_ylim(self.ego.y - 10, self.ego.y + 10)  # y축은 차량 중심 ±10m
        if real_time:
            plt.pause(0.01)
        
        plt.show()

    def is_done(self):
        self.done = False
        for obj in self.obs:
            if sat_collision_check(self.ego.rotated_corners, obj.rotated_corners):
                self.done = True
                self.collision_flag = True
                # print("Done with collision")
        if self.ego.x >= 200.0:
            self.done = True
            self.ep_end_flag = True
            # print("Done with end")



    def step(self, action):
        """
        action : 0, 1, 2 : 좌 중 우 
        """

        # print(f"action : {action}")
        action_1 = action[0] -1 
        action_2 = action[1]


        # 현 action을 따르는 거동이 가능한가? 



        # 전역 경로 생성
        self.map.make_global_path(self.ego, self.lane_order)
        
        # 로컬 경로 생성 
        path_x, path_y = self.map.make_local_path(self.ego, action_1, self.lane_order)
        
        
        car_x, car_y, car_heading, car_vx = self.ego.get_state()
        target_heading = get_target_heading(path_x, path_y)
        lateral_offset = np.abs(car_y - path_y[-1])
        heading_diff = np.abs(car_heading - target_heading)  # Heading 차이
        heading_diff_deg = np.degrees(heading_diff)  # degrees 단위로 변환
        
        if action_1 != 0 :
            if lateral_offset <= 0.3 and heading_diff_deg <= 10.0:
                # print(f"Previous Lane Order : {self.lane_order}")
                self.lane_order += action_1 
                # print(f"New Lane order : {self.lane_order}")
                action_1 = 0
                self.map.prev_action = 0
                self.map.make_global_path(self.ego,self.lane_order)
                path_x, path_y = self.map.make_local_path(self.ego, action_1, self.lane_order)

        # 차량 횡방향 제어 
        steering_angle = self.controller.get_steering_angle(car_x, car_y, car_heading, path_x, path_y)

        # action 2 : - 3, -1, 0 +1 +3 
        action_2_acc_dict = dict()
        action_2_acc_dict[0] = -3.0
        action_2_acc_dict[1] = -1.0
        action_2_acc_dict[2] = 0.0 
        action_2_acc_dict[3] = 1.0 
        action_2_acc_dict[4] = 3.0
        ego_spd = max(self.ego.vx + self.time_step * action_2_acc_dict[action_2], 0.0)
        # 차량 속도와 조향 각도로 차량 업데이트
        self.ego.update(speed=ego_spd, steering_angle=steering_angle)
        self.lpp_x = path_x
        self.lpp_y = path_y

        ####################### STATE, Reward #######################
        """
        STATE
        """
        obj_dict, dist_dict, ttc_dict = self.find_obs(self.obs,self.ego)

        self.state[0] = self.ego.vx
        self.state[1] = self.get_lane_for_object(self.ego.y)
        self.state[2] = self.lane_order
        self.state[3] = 100/3.6     # Spd Limit
        self.state[4] = self.prev_action
        self.state[5] = self.num_lanes  # 차선의 수 

        self.state[6] = ttc_dict["LF"]
        self.state[7] = ttc_dict["CF"]
        self.state[8] = ttc_dict["RF"]
        self.state[9] = ttc_dict["LR"]
        self.state[10] = ttc_dict["CR"]
        self.state[11] = ttc_dict["RR"]

        self.state[6] = dist_dict["LF"]
        self.state[7] = dist_dict["CF"]
        self.state[8] = dist_dict["RF"]
        self.state[9] = dist_dict["LR"]
        self.state[10] = dist_dict["CR"]
        self.state[11] = dist_dict["RR"]

        self.is_done()
        
        """
        Reward
        """
        self.reward = 1.0
        self.reward += self.ego.vx
        if self.collision_flag == True: 
            self.reward -= 20000.0
        if action_1 != self.prev_action:
            self.reward -= 0.2
        if self.ep_end_flag:
            self.reward += 2000.0
            
        """
        PRINT
        """
        print(f"Ego vx : {self.ego.vx:.3f}\tx : {self.ego.x:.3f}\ty: {self.ego.y:.3f}")

        

        ####################### Previous Step Update #######################
        self.prev_action = action_1
        


        self.state = np.array(self.state)
        
        # self.state = np.array(self.state)
        # self.state = np.array(self.state)
        return self.state, self.reward, self.done

    
def main():
    env = Env()
    while True : 
        action = np.random.choice([-1, 0, 1])
        start_time = time.time()
        state, reward, done = env.step(action)
        # env.visualize(real_time=True)
        if done:
            break

    print("END")
        

if __name__ == "__main__":
    main()