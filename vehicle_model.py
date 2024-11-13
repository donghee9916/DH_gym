import numpy as np

class BicycleModel:
    def __init__(self, x, y, heading, wheelbase, dt, max_steering_angle=np.pi/4):
        """
        :param x: 초기 x 위치
        :param y: 초기 y 위치
        :param heading: 초기 heading (radians)
        :param wheelbase: 차량의 휠베이스 (차량 앞 바퀴와 뒷 바퀴 사이의 거리)
        :param dt: 시간 간격 (초 단위)
        :param max_steering_angle: 최대 조향 각도 (radians), 기본값은 ±45도
        """
        self.x = x
        self.y = y
        self.heading = heading
        self.vx = 0.0/3.6  # 초기 속도
        self.steering_angle = 0  # 초기 조향 각도
        self.wheelbase = wheelbase  # 휠베이스
        self.dt = dt  # 시간 간격
        self.max_steering_angle = max_steering_angle  # 최대 조향 각도
        self.W = 1.825
        self.L = 4.650
        self.rotated_corners = []
        self.get_corners()


    def update(self, speed, steering_angle):
        """
        차량의 위치와 heading을 업데이트합니다.
        
        :param speed: 차량 속도 (m/s)
        :param steering_angle: 차량의 조향 각도 (radians)
        """
        # 조향 각도가 최대값을 넘지 않도록 제한
        steering_angle = np.clip(steering_angle, -self.max_steering_angle, self.max_steering_angle)

        # 차량 속도 업데이트
        self.vx = speed
        self.steering_angle = steering_angle
        
        # Bicycle 모델의 동역학 방정식
        # 차선의 이동 방향 계산
        self.x += self.vx * np.cos(self.heading) * self.dt
        self.y += self.vx * np.sin(self.heading) * self.dt
        
        # heading 업데이트
        self.heading += (self.vx / self.wheelbase) * np.tan(self.steering_angle) * self.dt

        self.get_corners()

    def get_state(self):
        """
        현재 차량의 상태를 반환합니다.
        
        :return: 차량의 위치 (x, y), heading (radians), 속도 (vx)
        """
        return self.x, self.y, self.heading, self.vx

    def set_position(self, x, y, heading):
        """
        차량의 위치와 heading을 설정합니다.
        
        :param x: x 위치
        :param y: y 위치
        :param heading: heading (radians)
        """
        self.x = x
        self.y = y
        self.heading = heading

    
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
