import numpy as np
import pygame
import gym
from gym import spaces
import random
import time
from .env_classes import * 
from Vehicle_Model.vehicle_model import BicycleModel
from Control.purepursuit import PurePursuitController
from utils import * 



class MultiAgentEnv(gym.Env):
    def __init__(self, state_dim, num_agents=2):
        super(MultiAgentEnv, self).__init__()

        self.num_agents = num_agents  # 에이전트의 수
        self.state_dim = state_dim
        self.pygame_flag = True
        pygame.init()
        self.screen_width = 1500
        self.screen_height = 500
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        self.clock = pygame.time.Clock()

        self.y_range_min = -10.0
        self.y_range_max = 10.0

        self.x_range_min = -50.0
        self.x_range_max = 100.0

        self.num_lanes = 3
        self.lane_width = 3

        self.map = Map(self.num_lanes, self.lane_width)

        self.time_step = 0.02
        self.vehicles = []

        # 차량 및 장애물 객체 생성
        for _ in range(self.num_agents):
            ego = BicycleModel(self.map.lanes[self.num_lanes//2+1][0][0], 
                               self.map.lanes[self.num_lanes//2+1][1][0],
                               heading=0.0, wheelbase=1.825, dt=self.time_step)
            self.vehicles.append(ego)
        
        self.controller = PurePursuitController(lookahead_distance=15.0, wheelbase=1.825)
        
        self.lane_order = self.num_lanes // 2 + 1 
        self.collision_flag = False
        self.done_flag = False

        self.prev_actions = [0] * self.num_agents
        self.state = np.zeros((self.num_agents, self.state_dim))

        self.obs = self.generate_static_obstacles()

    def reset(self):
        self.vehicles = []
        for _ in range(self.num_agents):
            ego = BicycleModel(self.map.lanes[self.num_lanes//2+1][0][0], 
                               self.map.lanes[self.num_lanes//2+1][1][0],
                               heading=0.0, wheelbase=1.825, dt=self.time_step)
            self.vehicles.append(ego)

        self.prev_actions = [0] * self.num_agents
        self.state = np.zeros((self.num_agents, self.state_dim))
        
        # 장애물 리셋
        self.obs = self.generate_static_obstacles()

        return self.state

    def step(self, actions):
        rewards = []
        done = False

        for i, ego in enumerate(self.vehicles):
            action_1 = actions[i][0] - 1 
            action_2 = actions[i][1]

            # 전역 경로 생성
            self.map.make_global_path(ego, self.lane_order)
            
            # 로컬 경로 생성
            path_x, path_y = self.map.make_local_path(ego, action_1, self.lane_order)
            
            car_x, car_y, car_heading, car_vx = ego.get_state()
            target_heading = get_target_heading(path_x, path_y)
            lateral_offset = np.abs(car_y - path_y[-1])
            heading_diff = np.abs(car_heading - target_heading)
            heading_diff_deg = np.degrees(heading_diff)
            
            if action_1 != 0:
                if lateral_offset <= 0.3 and heading_diff_deg <= 10.0:
                    self.lane_order += action_1
                    self.lane_order = max(min(self.lane_order, 3), 1)
                    action_1 = 0
                    self.map.prev_action = 0
                    self.map.make_global_path(ego, self.lane_order)
                    path_x, path_y = self.map.make_local_path(ego, action_1, self.lane_order)

            # 차량 횡방향 제어
            steering_angle = self.controller.get_steering_angle(car_x, car_y, car_heading, path_x, path_y)

            # 액션 2에 따른 속도 업데이트
            action_2_acc_dict = {0: -3.0, 1: -1.0, 2: 0.0, 3: 1.0, 4: 3.0}
            ego_spd = max(ego.vx + self.time_step * action_2_acc_dict[action_2], 0.0)
            
            # 차량 업데이트
            ego.update(speed=ego_spd, steering_angle=steering_angle)
            
            # 장애물과의 충돌 체크 및 보상 계산
            obj_dict, dist_dict, ttc_dict = self.find_obs(self.obs, ego)
            self.state[i][0] = ego.vx
            self.state[i][1] = self.get_lane_for_object(ego.y)
            self.state[i][2] = self.lane_order
            self.state[i][3] = 100/3.6  # Spd Limit
            self.state[i][4] = self.prev_actions[i]
            self.state[i][5] = self.num_lanes
            self.state[i][6] = dist_dict["LF"]
            self.state[i][7] = dist_dict["CF"]
            self.state[i][8] = dist_dict["RF"]
            self.state[i][9] = dist_dict["LR"]
            self.state[i][10] = dist_dict["CR"]
            self.state[i][11] = dist_dict["RR"]
            
            # 보상 계산
            reward = self.calculate_reward(ego, action_1, action_2, dist_dict, ttc_dict)
            rewards.append(reward)

            # 충돌 체크
            if self.is_done():
                done = True
                break
        
        return self.state, rewards, done

    def is_done(self):
        for ego in self.vehicles:
            if any(sat_collision_check(ego.rotated_corners, obj.rotated_corners) for obj in self.obs):
                self.collision_flag = True
                return True
            if ego.x >= 200.0:
                return True
        return False

    def generate_static_obstacles(self, num_obstacles=20):
        obstacles = []
        lane_orders = [1, 2, 3]
        id = 0
        lane_index = 0
        obj_x = 0.0
        while obj_x <= 500:
            id += 1
            lane_order = lane_orders[lane_index]
            lane_y = self.map.lanes[lane_order][1][0]
            obj_x += 50.0
            vx = 0.0
            vy = 0.0
            is_moving = False
            obstacle = Object(id=id, x=obj_x, y=lane_y, vx=vx, vy=vy, lane_order=lane_order, time_step=self.time_step, is_moving=is_moving)
            obstacles.append(obstacle)
            lane_index += 1
            if lane_index >= 3:
                lane_index -= 3
        return obstacles

    def find_obs(self, obs, ego):
        ego_obj_dict = {"LF": [], "CF": [], "RF": [], "LR": [], "CR": [], "RR": []}
        ego_dist_dict = {"LF": float(999.0), "CF": float(999.0), "RF": float(999.0), "LR": float(999.0), "CR": float(999.0), "RR": float(999.0)}
        ego_ttc_dict = {"LF": float(999.0), "CF": float(999.0), "RF": float(999.0), "LR": float(999.0), "CR": float(999.0), "RR": float(999.0)}

        lane_offset = {self.lane_order-1: ["LF", "LR"], self.lane_order: ["CF", "CR"], self.lane_order+1: ["RF", "RR"]}

        for obj in obs:
            lane_id = self.get_lane_for_object(obj.y)
            if lane_id in lane_offset:
                if obj.x >= ego.x:
                    lane = lane_offset[lane_id][0]  # Front (LF, CF, RF)
                    ego_obj_dict[lane].append(obj)
                    dist = np.sqrt((ego.x - obj.x)**2 + (ego.y - obj.y)**2)
                    if dist < ego_dist_dict[lane]:
                        ego_dist_dict[lane] = dist
                        ego_ttc_dict[lane] = dist / ego.vx if ego.vx > 0 else 999.0
                else:
                    lane = lane_offset[lane_id][1]  # Rear (LR, CR, RR)
                    ego_obj_dict[lane].append(obj)
                    dist = np.sqrt((ego.x - obj.x)**2 + (ego.y - obj.y)**2)
                    if dist < ego_dist_dict[lane]:
                        ego_dist_dict[lane] = dist
                        ego_ttc_dict[lane] = dist / ego.vx if ego.vx > 0 else 999.0
        return ego_obj_dict, ego_dist_dict, ego_ttc_dict

    def calculate_reward(self, ego, action_1, action_2, dist_dict, ttc_dict):
        reward = 0
        if self.collision_flag:
            reward = -100
        else:
            reward += 0.1 * ego.vx  # 속도에 따른 보상
            if ttc_dict["LF"] < 5.0:
                reward -= 1.0  # 전방 장애물과의 충돌 위험이 높을 때 패널티
        return reward

    def get_lane_for_object(self, obj_y):
        return 1 if obj_y < self.map.lanes[1][1][0] else 2 if obj_y < self.map.lanes[2][1][0] else 3

    def render(self):
        self.screen.fill((0, 0, 0))
        self.map.render(self.screen)

        for ego in self.vehicles:
            ego.render(self.screen)

        for obstacle in self.obs:
            obstacle.render(self.screen)
        
        pygame.display.flip()

    def close(self):
        pygame.quit()
