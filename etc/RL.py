import numpy as np
import matplotlib.pyplot as plt
from utils import * 


# 차량의 속도 (kph -> m/s로 변환)
speed_kph = 30.0  # 차량 속도 (kph)
speed_mps = speed_kph * 1000 / 3600  # m/s

# 시간 단계
time_step = 0.02  # 20 ms

# 3개의 평행한 차선 (y = -1, 0, 1)
lane_y_positions = np.array([-3, 0, 3])  # 3개의 차선 위치



# GPP 경로 (직선 경로 예시)
gpp_x = np.linspace(0, 100, 100)  # x 좌표 (0에서 100m까지)
gpp_y = np.zeros_like(gpp_x)  # y 좌표 (0에서 0m까지, 차량이 기본적으로 중간 차선에서 주행)
gpp_z = np.zeros_like(gpp_x)  # z 좌표 (2D 경로, z는 0으로 설정)

# 차량 및 장애물 객체 생성
ego = Ego(gpp_x, lane_y_positions, time_step, speed_mps)
obs_1 = Object(gpp_x, gpp_y, is_moving=False)

# GPP 경로 시각화
fig, ax = plt.subplots(figsize=(10, 6))

# 3개의 차선 시각화
for lane_y in lane_y_positions:
    ax.plot(gpp_x, np.full_like(gpp_x, lane_y), linestyle='--', label=f'Lane {lane_y}')

# 차량 경로 시각화
path_x = [ego.x]
path_y = [ego.y]

# 실시간 시뮬레이션
plt.ion()  # Interactive mode 활성화
plt.show()

while True:
    # 차량 위치 업데이트
    ego.update()
    
    # 차량과 장애물의 4개 코너 좌표 계산
    ego_corners = ego.get_corners()
    obs_1_corners = obs_1.get_corners()
    
    # SAT를 통해 충돌 판별
    if sat_collision_check(ego_corners, obs_1_corners):
        print(f"Collision detected at x={ego.x}, y={ego.y}")
    
    # 차량 이동 경로 기록
    path_x.append(ego.x)
    path_y.append(ego.y)

    # 시각화 업데이트
    ax.cla()  # 이전 플롯 지우기
    for lane_y in lane_y_positions:
        ax.plot(gpp_x, np.full_like(gpp_x, lane_y), linestyle='--', label=f'Lane {lane_y}')
    ax.scatter(obs_1.x, obs_1.y, color='red', s=100, label='Obstacle')  # 장애물
    ax.plot(path_x, path_y, label='Vehicle Path', color='green')  # 차량 경로
    ax.scatter(ego.x, ego.y, color='black')  # 현재 차량 위치 표시
    ax.legend()

    # 차량 중심을 기준으로 범위 설정
    ax.set_xlim(ego.x - 30, ego.x + 30)  # x축은 차량 중심 ±30m
    ax.set_ylim(ego.y - 10, ego.y + 10)  # y축은 차량 중심 ±10m

    plt.draw()
    plt.pause(0.01)  # 잠깐 대기

    # 경로 끝에 도달하면 종료
    if ego.x >= gpp_x[-1]:
        break

plt.ioff()  # Interactive mode 비활성화
plt.show()  # 마지막 그래프 보여주기