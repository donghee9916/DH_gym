import numpy as np
from utils import * 
class Ego:
    def __init__(self, gpp_x, gpp_y, dt, speed):
        print(f"gpp type : {type(gpp_x)}")
        print(f"gpp type : {type(gpp_y)}")
        print(f"gpp x : {gpp_x}")
        print(f"gpp y : {gpp_y}")
        self.x = gpp_x
        self.y = gpp_y
        self.z = 0.0
        self.heading = 0.0
        self.dt = dt
        self.vx = speed
        self.vy = 0.0     

        self.W = 1.825
        self.L = 4.650

        self.rotated_corners = []
        self.get_corners()

        self.lane_order = 2

    def update(self):
        self.x += self.vx * self.dt * np.cos(self.heading)  # x 방향 업데이트
        self.y += self.vx * self.dt * np.sin(self.heading)  # y 방향 업데이트
        self.get_corners()
    
    
    def get_corners(self):
        cos_heading = np.cos(self.heading)
        sin_heading = np.sin(self.heading)
        # 차량의 4개 코너 (왼쪽 앞, 오른쪽 앞, 오른쪽 뒤, 왼쪽 뒤)
        corners = np.array([
            [self.L / 2, self.W / 2],  # 왼쪽 앞 (FL)
            [self.L / 2, -self.W / 2], # 오른쪽 앞 (FR)
            [-self.L / 2, -self.W / 2],# 오른쪽 뒤 (RR)
            [-self.L / 2, self.W / 2]  # 왼쪽 뒤 (RL)
        ])
        rotation_matrix = np.array([
            [cos_heading, -sin_heading],
            [sin_heading, cos_heading]
        ])

        # 회전 적용 후, 차량 중심으로 이동
        rotated_corners = np.dot(corners, rotation_matrix.T)  # 회전 후 코너 좌표 계산
        # 회전된 좌표를 차량의 중심 (self.x, self.y)으로 이동
        rotated_corners[:, 0] += self.x  # x 좌표 이동
        rotated_corners[:, 1] += self.y  # y 좌표 이동
        
        self.rotated_corners = rotated_corners


class Object:
    def __init__(self,id, x, y,vx,vy,lane_order, time_step = 0.02,is_moving=False):
        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy= vy
        self.heading = 0.0
        self.time_step = time_step
        self.is_moving = is_moving

        self.W = 1.825
        self.L = 4.650
        self.lane_order = lane_order 
        self.rotated_corners = []
        self.get_corners()

    def update(self):
        if self.is_moving:
            self.x += self.vx * self.time_step
            self.y += self.vy * self.time_step
        self.get_corners()


    def get_corners(self):
        # 장애물의 4개 코너 (사각형 형태로 가정)
        cos_heading = np.cos(self.heading)
        sin_heading = np.sin(self.heading)
        
        # 차량 회전 고려
        rotation_matrix = np.array([
            [cos_heading, -sin_heading],
            [sin_heading, cos_heading]
        ])
        corners = np.array([
            [self.L / 2, self.W / 2],  # 왼쪽 앞 (FL)
            [self.L / 2, -self.W / 2], # 오른쪽 앞 (FR)
            [-self.L / 2, -self.W / 2],# 오른쪽 뒤 (RR)
            [-self.L / 2, self.W / 2]  # 왼쪽 뒤 (RL)
        ])
        rotation_matrix = np.array([
            [cos_heading, -sin_heading],
            [sin_heading, cos_heading]
        ])

        # 회전 적용 후, 차량 중심으로 이동
        rotated_corners = np.dot(corners, rotation_matrix.T)  # 회전 후 코너 좌표 계산
        # 회전된 좌표를 차량의 중심 (self.x, self.y)으로 이동
        rotated_corners[:, 0] += self.x  # x 좌표 이동
        rotated_corners[:, 1] += self.y  # y 좌표 이동
        self.rotated_corners = rotated_corners


    
class Map:
    def __init__(self, num_lanes, lane_width):
        """
        :param num_lanes: 차선의 수
        :param lane_width: 차선 폭 (m)
        :param x_points: x좌표의 점들 (경로의 x좌표들, 예: np.linspace로 생성)
        """
        self.num_lanes = num_lanes
        self.lane_width = lane_width
        self.global_path = dict()
        total_length = 1000 # 경로 길이 [m]
        self.x = np.linspace(0, total_length, total_length * 2)

        self.minimum_lane_length = 70.0 # [m] 최소 차선 길이 
        self.lc_dist =20.0 # [m] 차선 변경 길이


        self.prev_action = 0
        self.prev_x = None
        self.prev_y = None 

        
        # y 좌표 생성: 각 차선은 lane_width만큼 차이가 나므로, 
        # 중앙 차선이 y=0에 위치하고 그 위아래로 차선들이 배치됩니다.
        self.y = np.linspace(-(num_lanes // 2) * lane_width, (num_lanes // 2) * lane_width, num_lanes)
        # 각 차선별 x, y 좌표 생성
        self.lanes = {}  # 딕셔너리로 차선 정보를 저장할 예정
        for i, y in enumerate(self.y):
            lane_id = i + 1  # lane_1, lane_2, ...
            self.lanes[lane_id] = [self.x, np.full_like(self.x, y)]

        self.lpp_x = self.lanes[num_lanes//2 + 1][0] 
        self.lpp_y = self.lanes[num_lanes//2 + 1][1]

    def make_global_path(self,ego, lane_order):
        for key in self.lanes:
            if key == lane_order:
                # 현재 차선의 경로
                self.global_path[0] = self.get_lane_path(ego, self.lanes[key])
                
                # 좌측 차선 경로 (기존 차선이 1번 이상일 경우)
                if key > 1:
                    self.global_path[-1] = self.get_lane_path(ego, self.lanes[key - 1])
                else:
                    # print("No More Left Lane")
                    self.global_path[-1] =  self.get_lane_path(ego, self.lanes[key])  # 좌측 차선이 없으면 None

                # 우측 차선 경로 (기존 차선이 마지막 차선 미만일 경우)
                if key < self.num_lanes:
                    self.global_path[1] = self.get_lane_path(ego, self.lanes[key + 1])
                else:
                    # print("No More Right Lane")
                    self.global_path[1] = self.get_lane_path(ego, self.lanes[key])  # 우측 차선이 없으면 None

    def get_lane_path(self, ego, lane_data):
        """
        주어진 차선에 대해 차량의 현재 위치를 기준으로 최소 100m 이상의 경로를 생성
        """
        lane_x, lane_y = lane_data
        start_idx = int(np.argmin(np.abs(lane_x - ego.x)))  # ego 차량과 가장 가까운 x값 인덱스 찾기
        # 경로 길이가 100m 이상이 되도록 인덱스 계산
        end_idx = int(start_idx + self.minimum_lane_length * 8) 
        if end_idx >= len(lane_x):
            end_idx = len(lane_x) - 1  # 끝까지 다 사용하도록 처리
            print("global path end")


        # 경로 반환
        return lane_x[start_idx:end_idx], lane_y[start_idx:end_idx]

    def make_local_path(self, ego, action, lane_order):
        lane_change_dist = 20 # [m] 차선 변경 길이 
        point_dist = 0.5 
        if self.prev_action != action or self.prev_x == None:
            # 주어진 action에 해당하는 차선이 있다면
            action = min(max(-1, action),1)

            if self.global_path.get(action) is not None:
                # 글로벌 경로에서 현재 선택된 차선 경로 가져오기
                global_x, global_y = self.global_path[action]
                
                # print(f"action : {action}, lane order : {lane_order}, g_x size : {len(global_x)}, g_y size : {len(global_y)}, ego x: {ego.x}, ego y: {ego.y}")
                start_idx = np.argmin(np.abs(np.sqrt((global_x - ego.x)**2 +  (global_y - ego.y)**2)))  # ego 차량과 가장 가까운 x 인덱스 찾기

                # 40m 길이의 로컬 경로를 생성
                end_idx = start_idx + int(lane_change_dist / point_dist)  # 40m 뒤의 인덱스
                # end_idx가 배열 길이를 벗어나지 않도록 수정
                end_idx = min(end_idx, len(global_x) - 1)
                end_point_x = global_x[end_idx]
                end_point_y = global_y[end_idx]

                # ego의 위치와 end_point의 좌표 사이를 bezier 커브로 잇는다.
                
                control_1_x = ego.x + self.lc_dist /3 * np.cos(ego.heading)
                control_1_y = ego.y + self.lc_dist /3 * np.sin(ego.heading)
                
                control_2_x = end_point_x + self.lc_dist/3 * np.cos(np.pi)
                control_2_y = end_point_y + self.lc_dist/3 * np.sin(np.pi)
                
                lc_x, lc_y = third_order_bezier_curve(ego.x, ego.y, control_1_x, control_1_y, control_2_x, control_2_y, end_point_x, end_point_y)
                
                # 추가적인 경로 길이를 계산하여 붙여넣기
                additional_length = int((self.minimum_lane_length - self.lc_dist) * 2)
                end_idx = min(end_idx + additional_length, len(global_x) - 1)
                # 추가 경로 연결
                lc_x.extend(global_x[end_idx: end_idx + additional_length])
                lc_y.extend(global_y[end_idx: end_idx + additional_length])
                self.prev_x = lc_x
                self.prev_y = lc_y
                self.prev_action = action
                

                self.lpp_x = lc_x 
                self.lpp_y = lc_y
                return lc_x, lc_y
            
            else :       
                return self.lpp_x, self.lpp_y
        else:

            global_x, global_y = self.lanes[max(min(lane_order + action,1),self.num_lanes)]
            # global_x, global_y = self.global_path[action]
            self.prev_action = action
            current_index = int(np.argmin(np.abs(np.sqrt((self.prev_x - ego.x)**2 +  (self.prev_y - ego.y)**2))))
            self.prev_x = self.prev_x[current_index:]
            self.prev_y = self.prev_y[current_index:]
            

            global_path_index = int(np.argmin(np.abs(np.sqrt((global_x - self.prev_x[-1])**2 +  (global_y - self.prev_y[-1])**2))))
            # print(f"current : {current_index}, global : {global_path_index}")
            # print(f"Local x: {self.prev_x[-1]}, y: {self.prev_y[-1]}, Global x: {global_x[global_path_index]}, y: {global_y[global_path_index]}")
            # print(f"prev x size : {len(self.prev_x)}, y size : {len(self.prev_y)}")
            # print(f"min : {global_path_index}, max : {global_path_index + int(self.minimum_lane_length * 2) - len(self.prev_x)}")
            add_x = global_x[global_path_index: global_path_index + int(self.minimum_lane_length * 2) - len(self.prev_x) ]
            add_y = global_y[global_path_index: global_path_index + int(self.minimum_lane_length * 2) - len(self.prev_y) ]
            self.prev_x.extend(global_x[global_path_index: global_path_index + int(self.minimum_lane_length * 2) - len(self.prev_x) ])
            self.prev_y.extend(global_y[global_path_index: global_path_index + int(self.minimum_lane_length * 2) - len(self.prev_y) ])
            
            self.lpp_x = self.prev_x
            self.lpp_y = self.prev_y

            return self.prev_x, self.prev_y
            




