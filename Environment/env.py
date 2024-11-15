import numpy as np
import matplotlib.pyplot as plt
from Vehicle_Model.vehicle_model import BicycleModel
from Control.purepursuit import PurePursuitController
from utils import *  # 충돌 체크 함수 및 Bezier 관련 함수들
from .env_classes import *
import random
import time
import math 
import cv2 
import os 

import pygame
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


        # self.video_writer = None
        # self.video_output_dir = "video_output"
        # # 비디오 저장 폴더가 없으면 생성
        # if not os.path.exists(self.video_output_dir):
        #     os.makedirs(self.video_output_dir)
        # self.fps = 30
        # self.pygame_flag = True # 종료 : False , 켜져있음 : True'
        # self.episode_num = 1
        # pygame.init()
        # self.screen_width = 1500
        # self.screen_height = 500
        # self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        # self.clock = pygame.time.Clock()    # FPS 제어 

        # self.font = pygame.font.Font(None, 24)  # 텍스트 크기 설정 (예: 24픽셀)
        # self.text_color = (0, 0, 0)  # 텍스트 색상 (흰색)

        # # Y축 범위 설정 (중앙: 0, 범위: +10 ~ -10)
        # self.y_range_min = -10.0
        # self.y_range_max = 10.0

        # # X축 범위 설정 (중앙: Ego 기준, 범위: -50 ~ +100)
        # self.x_range_min = - 50.0  # Ego 기준 좌측 50m
        # self.x_range_max = + 100.0  # Ego 기준 우측 100m


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
        self.prev_action = 1

        self.lpp_x = []
        self.lpp_y = []



        # self.visualize(real_time=False)

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
        while obj_x <= 500:
            id +=1 
            lane_order = lane_orders[lane_index]
            lane_y = self.map.lanes[lane_order][1][0]
            obj_x += 50.0 
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
    def clamp(self, value, min_value, max_value):
        return max(min(value, max_value),min_value)
    
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
                    ego_ttc_dict[lane] =self.clamp(dist / (ego.vx - obj.vx), -30.0, 30.0)

        return ego_obj_dict, ego_dist_dict, ego_ttc_dict


    def render(self):
        """
        Real Time Visualization With pygame
        """
        # if self.pygame_flag == False:
        #     self.restart()
        #     self.pygame_flag = True
        # Fill screen with white
        # self.screen.fill((255, 255, 255))

        # EGO 차량 그리기
        ego_rect_x = [corner[0] for corner in self.ego.rotated_corners] + [self.ego.rotated_corners[0][0]]
        ego_rect_y = [corner[1] for corner in self.ego.rotated_corners] + [self.ego.rotated_corners[0][1]]

        # 좌표를 화면 크기에 맞게 변환 (비율 맞추기)
        ego_rect_x = [self.scale_to_screen(x, self.ego.y)[0] for x in ego_rect_x]
        ego_rect_y = [self.scale_to_screen(self.ego.x, y)[1] for y in ego_rect_y]

        # 차량 그리기 (파란색)
        pygame.draw.polygon(self.screen, (0, 0, 255), list(zip(ego_rect_x, ego_rect_y)))
        pygame.draw.circle(self.screen, (0, 0, 0), (int(self.scale_to_screen(self.ego.x, self.ego.y)[0]), 
                                                     int(self.scale_to_screen(self.ego.x, self.ego.y)[1])), 5)

        # State Text 작성 
        text_surface = self.font.render(f"Speed: {self.state[0]:.2f} m/s", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x, self.ego.y+1)))
        text_surface = self.font.render(f"Lane Order: {self.state[2]}", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x, self.ego.y+1.5)))
        
        # Dist 
        text_surface = self.font.render(f"TTC FL: {self.state[6]:.2f} m", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x+20, -3)))
        text_surface = self.font.render(f"TTC FC: {self.state[7]:.2f} m", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x+20, 0.0)))
        text_surface = self.font.render(f"TTC FR: {self.state[8]:.2f} m", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x+20, +3)))
        text_surface = self.font.render(f"TTC RL: {self.state[9]:.2f} m", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x-20, -3)))
        text_surface = self.font.render(f"TTC RC: {self.state[10]:.2f} m", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x-20, 0.0)))
        text_surface = self.font.render(f"TTC RR: {self.state[11]:.2f} m", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x-20, +3)))

        # TTC
        text_surface = self.font.render(f"Dist FL: {self.state[12]:.2f} sec", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x+20, -1.5)))
        text_surface = self.font.render(f"Dist FC: {self.state[13]:.2f} sec", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x+20, +1.5)))
        text_surface = self.font.render(f"Dist FR: {self.state[14]:.2f} sec", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x+20, +4.5)))
        text_surface = self.font.render(f"Dist RL: {self.state[15]:.2f} sec", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x-20, -1.5)))
        text_surface = self.font.render(f"Dist RC: {self.state[16]:.2f} sec", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x-20, +1.5)))
        text_surface = self.font.render(f"Dist RR: {self.state[17]:.2f} sec", True, self.text_color)
        self.screen.blit(text_surface, (self.scale_to_screen(self.ego.x-20, +4.5)))

        
        # Episode 번호 텍스트 표시
        episode_text = f"Episode: {self.episode_num}"  # 현재 episode 번호를 표시
        font = pygame.font.Font(None, 36)  # 폰트 크기 설정
        text_surface = font.render(episode_text, True, (0, 0, 0))  # 텍스트 표기 (검정색)

        # 화면 중앙 상단 위치 계산
        text_width = text_surface.get_width()  # 텍스트 너비
        text_height = text_surface.get_height()  # 텍스트 높이
        screen_center_x = self.screen_width // 2  # 화면 중앙 X 좌표
        text_x = screen_center_x - text_width // 2  # 텍스트가 화면 중앙에 오도록 X 좌표 조정
        text_y = 10  # 상단에서 10픽셀 정도 떨어진 위치

        # 텍스트 화면에 그리기
        self.screen.blit(text_surface, (text_x, text_y))



        # 장애물 그리기
        for obj in self.obs:
            lane_id = self.get_lane_for_object(obj.y)

            if lane_id == self.lane_order - 1:
                color = (255, 0, 0) if obj.x >= self.ego.x else (169, 169, 169)
            elif lane_id == self.lane_order:
                color = (255, 0, 0) if obj.x >= self.ego.x else (169, 169, 169)
            elif lane_id == self.lane_order + 1:
                color = (255, 0, 0) if obj.x >= self.ego.x else (169, 169, 169)
            else:
                color = (0, 0, 0)  # 기본 색상 (검정)

            # 장애물 사각형 그리기
            obs_rect_x = [corner[0] for corner in obj.rotated_corners] + [obj.rotated_corners[0][0]]
            obs_rect_y = [corner[1] for corner in obj.rotated_corners] + [obj.rotated_corners[0][1]]

            obs_rect_x = [self.scale_to_screen(x, obj.y)[0] for x in obs_rect_x]
            obs_rect_y = [self.scale_to_screen(obj.x, y)[1] for y in obs_rect_y]

            pygame.draw.polygon(self.screen, color, list(zip(obs_rect_x, obs_rect_y)))
            pygame.draw.circle(self.screen, color, (int(self.scale_to_screen(obj.x, obj.y)[0]), 
                                                     int(self.scale_to_screen(obj.x, obj.y)[1])), 5)
            font = pygame.font.Font(None, 36)  # 폰트 크기 설정
            lane_text = f"Lane: {lane_id}"  # 차선 정보
            text_surface = font.render(lane_text, True, (0, 0, 0))  # 텍스트 표기 (검정색)
            
            # 텍스트 위치 설정: 장애물 중심 근처에 표시
            text_pos = (int(self.scale_to_screen(obj.x+5, obj.y+2)[0] + 10), int(self.scale_to_screen(obj.x, obj.y)[1] - 10))
            # 텍스트 화면에 그리기
            self.screen.blit(text_surface, text_pos)

        # 차선 표시
        if len(self.lpp_x) > 1 and len(self.lpp_y) > 1:  # 리스트에 점이 두 개 이상 있는지 확인
            pygame.draw.lines(self.screen, (0, 255, 0), False, 
                              [(self.scale_to_screen(x, y)) for x, y in zip(self.lpp_x, self.lpp_y)], 3)

        for lane_id, (x, y) in self.map.lanes.items():
            pygame.draw.lines(self.screen, (0, 0, 0), False, 
                              [(self.scale_to_screen(xx, yy)) for xx, yy in zip(x, y)], 2)

        # 화면 업데이트
        pygame.display.flip()

        self.clock.tick(30)  # FPS 조정 (30 FPS로 설정)

        # # 50 episode마다 비디오 저장
        # if self.episode_num-1 % 50 == 0:
        #     self.save_frame_as_video()

    def start_video_recording(self):
        """비디오 기록을 시작하는 함수"""
        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # XVID 또는 다른 코덱을 사용할 수 있습니다.
        video_file_name = os.path.join(self.video_output_dir, f"episode_{self.episode_num}.avi")
        self.video_writer = cv2.VideoWriter(video_file_name, fourcc, self.fps, (self.screen_width, self.screen_height))

    def save_frame_as_video(self):
        """현재 Pygame 화면을 비디오 파일로 저장하는 함수"""
        if self.video_writer is None:
            self.start_video_recording()  # 비디오 기록이 시작되지 않았다면 시작
        
        # Pygame 화면을 OpenCV에서 사용할 수 있는 포맷으로 변환
        frame = pygame.surfarray.array3d(self.screen)  # (height, width, 3)
        frame = cv2.transpose(frame)  # Pygame은 (width, height)로, OpenCV는 (height, width)로 처리합니다.

        # 프레임을 비디오 파일에 추가
        self.video_writer.write(frame)

    def scale_to_screen(self, x, y):
        """환경 좌표를 화면 픽셀 좌표로 변환"""
         # X 좌표 변환: ego.x 기준으로 -50m ~ +100m를 0 ~ 화면 폭(800)으로 변환
        scaled_x = int(((x - self.ego.x+50) / (150.0)) * self.screen_width)

        # Y 좌표 변환: +10m ~ -10m를 0 ~ 화면 높이(600)로 변환
        scaled_y = int(((y + 10.0) / (20.0)) * self.screen_height)

        return scaled_x, scaled_y


    def close(self):
        """시뮬레이션 종료 시 pygame 창을 닫을 때 호출"""
        self.pygame_flag = False
        pygame.quit()
        pygame.display.quit()

    def restart(self):
        pygame.init()
        # 새 화면 설정
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        # 화면 업데이트를 위한 기본 설정
        self.clock = pygame.time.Clock()
            
    
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
        if self.ego.x >= 500.0:
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
            if lateral_offset <= 0.5 or heading_diff_deg <= 10.0:
                # print(f"Previous Lane Order : {self.lane_order}")
                self.lane_order += action_1 
                self.lane_order = max(min(self.lane_order, 3),1)
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

        self.state[12] = dist_dict["LF"]
        self.state[13] = dist_dict["CF"]
        self.state[14] = dist_dict["RF"]
        self.state[15] = dist_dict["LR"]
        self.state[16] = dist_dict["CR"]
        self.state[17] = dist_dict["RR"]

        self.is_done()
        
        """
        Reward
        """
        minimum_spd = 30/3.6
        maximum_spd = 100/3.6
        self.reward = 100.0

        # 8초간의 영역에 대해 weight를 넣겠다. 
        collision_threshold_time = 8
        dist_threshold = self.ego.vx * collision_threshold_time 

        # Condition 1 : dist가 5도 안남았는데, 그 차선을 선택하는 행동을 한다?
        if action_1 == -1 and dist_dict["LF"] <=dist_threshold:
            self.reward -=dist_threshold - dist_dict["LF"]
        elif action_1 == 0 and dist_dict["CF"] <=dist_threshold:
            self.reward -=dist_threshold - dist_dict["CF"]
        elif action_1 == 1 and dist_dict["RF"] <=dist_threshold:
            self.reward -=dist_threshold - dist_dict["RF"]

        if action_1 == -1 and ttc_dict["LF"] <= collision_threshold_time:
            self.reward -= collision_threshold_time - ttc_dict["LF"]
        elif action_1 == 0 and ttc_dict["CF"] <= collision_threshold_time:
            self.reward -= collision_threshold_time - ttc_dict["CF"]
        elif action_1 == 1 and ttc_dict["RF"] <= collision_threshold_time:
            self.reward -= collision_threshold_time - ttc_dict["RF"]

        self.reward += self.ego.x/ 10
        
        if self.lane_order == 1 and action_1 == -1:
            self.reward -= 200.0
        elif self.lane_order == 3 and action_1 == 1:
            self.reward -= 200.0
        


        if self.ego.vx < minimum_spd:
            self.reward -= (minimum_spd - self.ego.vx) / minimum_spd *2
        elif self.ego.vx < maximum_spd:
            self.reward += self.ego.vx/100 *2
        else:
            self.reward -= 1.0

        self.reward += self.ego.vx
        # if self.collision_flag == True: 
        #     self.reward -= 20000.0
        if action_1 != self.prev_action:
            self.reward -= 2.0

        if self.ego.vx == 0.0 and action_2_acc_dict[action_2] < 0:
            self.reward -= 10.0
        if self.ep_end_flag:
            self.reward += 2000.0
            
        """
        PRINT
        """
        # print(f"Ego vx : {self.ego.vx:.3f}\tx : {self.ego.x:.3f}\ty: {self.ego.y:.3f}\taction 1: {action_1}\taction 2: {action_2}")

        

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