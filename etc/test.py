import numpy as np
import matplotlib.pyplot as plt

def get_corners(x, y, L, W, heading):
    """
    차량의 4개의 코너를 계산하는 함수 (heading에 따른 회전 포함).
    x, y: 차량의 중심 좌표
    L: 차량의 길이
    W: 차량의 너비
    heading: 차량의 회전 각도 (라디안 단위)
    """
    cos_heading = np.cos(heading)
    sin_heading = np.sin(heading)
    
    # 차량의 4개 코너 (왼쪽 앞, 오른쪽 앞, 오른쪽 뒤, 왼쪽 뒤)
    corners = np.array([
        [L / 2, W / 2],  # 왼쪽 앞 (FL)
        [L / 2, -W / 2], # 오른쪽 앞 (FR)
        [-L / 2, -W / 2],# 오른쪽 뒤 (RR)
        [-L / 2, W / 2]  # 왼쪽 뒤 (RL)
    ])
    
    # 회전 행렬
    rotation_matrix = np.array([
        [cos_heading, -sin_heading],
        [sin_heading, cos_heading]
    ])
    
    # 회전된 코너 좌표 계산
    rotated_corners = np.dot(corners, rotation_matrix.T)
    
    # 차량 중심 (x, y)으로 이동
    rotated_corners[:, 0] += x  # x 좌표 이동
    rotated_corners[:, 1] += y  # y 좌표 이동
    
    return rotated_corners

def plot_vehicle(x, y, L, W):
    
    """
    회전하는 차량의 사각형을 계속해서 플로팅하는 함수
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)
    ax.set_aspect('equal', 'box')  # x, y 비율을 같게 설정

    # heading을 1도씩 증가시키며 시각화
    for heading_deg in range(0, 360, 1):  # 0도에서 360도까지, 1도씩 증가
        ax.cla()

        heading_rad = np.radians(-heading_deg)  # 각도를 라디안으로 변환
        
        # 차량의 4개 코너 계산
        corners = get_corners(x, y, L, W, heading_rad)
        
        # 차량 사각형 시각화
        ego_rect_x = [corner[0] for corner in corners] + [corners[0][0]]
        ego_rect_y = [corner[1] for corner in corners] + [corners[0][1]]
        
        ax.plot(ego_rect_x, ego_rect_y, color='blue', label="Ego Vehicle")
        plt.pause(0.05)  # 50ms 대기 시간 (애니메이션 효과)

    plt.show()

if __name__ == "__main__":
    x, y = 0, 0  # 차량의 중심 좌표
    L = 1.825     # 차량의 길이 (m)
    W = 4.650     # 차량의 너비 (m)
    
    plot_vehicle(x, y, L, W)
