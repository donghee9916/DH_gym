import numpy as np 
class PurePursuitController:
    def __init__(self, lookahead_distance, wheelbase, max_steering_angle=np.pi/4):
        """
        :param lookahead_distance: 목표 지점까지의 거리
        :param wheelbase: 차량의 휠베이스
        :param max_steering_angle: 최대 조향 각도
        """
        self.lookahead_distance = lookahead_distance
        self.wheelbase = wheelbase
        self.max_steering_angle = max_steering_angle

    def get_steering_angle(self, car_x, car_y, car_heading, path_x, path_y):
        """
        Pure Pursuit 제어기를 사용하여 차량의 조향 각도를 계산합니다.
        
        :param car_x: 차량의 현재 x 위치
        :param car_y: 차량의 현재 y 위치
        :param car_heading: 차량의 현재 heading (radians)
        :param path_x: 경로의 x 좌표 (목표 지점들)
        :param path_y: 경로의 y 좌표 (목표 지점들)
        :return: 차량의 조향 각도
        """
        # 차량과 경로 상의 각 지점 간의 거리 계산
        min_dist = float('inf')
        goal_idx = -1

        # 목표 지점 찾기 (lookahead_distance 만큼 앞의 지점)
        for i in range(len(path_x)):
            dist = np.sqrt((path_x[i] - car_x) ** 2 + (path_y[i] - car_y) ** 2)
            if dist > self.lookahead_distance and dist < min_dist:
                min_dist = dist
                goal_idx = i
        
        if goal_idx == -1:
            # 목표 지점이 없다면 현재 위치에서 가장 가까운 지점으로 설정
            goal_idx = len(path_x) - 1

        goal_x = path_x[goal_idx]
        goal_y = path_y[goal_idx]

        # 목표 지점과 차량의 상대 위치 계산
        delta_x = goal_x - car_x
        delta_y = goal_y - car_y

        # 목표 지점에 대한 차량의 조향각 계산
        alpha = np.arctan2(delta_y, delta_x) - car_heading
        steering_angle = np.arctan2(2 * self.wheelbase * np.sin(alpha) / self.lookahead_distance, 1)

        # 조향각을 최대값으로 제한
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)
        
        return steering_angle