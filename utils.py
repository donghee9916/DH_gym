import numpy as np 

def get_target_heading(path_x, path_y):
    """
    목표 차선의 heading을 계산하는 함수.
    이 예시에서는 마지막 두 점을 사용하여 heading을 구합니다.
    """
    dx = path_x[-1] - path_x[-2]  # 마지막 두 점의 x 차이
    dy = path_y[-1] - path_y[-2]  # 마지막 두 점의 y 차이
    target_heading = np.arctan2(dy, dx)  # atan2 함수로 heading 계산
    return target_heading


def bezier_curve(P0, P1, P2, P3, t):
    # 베지어 커브 계산 (Cubic Bezier curve)
    # return (1 - t)**3 * P0 + 3 * (1 - t)**2 * t * P1 + 3 * (1 - t) * t**2 * P2 + t**3 * P3
     # 각 점에 대해 x, y 값을 계산
    Bx = (1 - t)**3 * P0[0] + 3 * (1 - t)**2 * t * P1[0] + 3 * (1 - t) * t**2 * P2[0] + t**3 * P3[0]
    By = (1 - t)**3 * P0[1] + 3 * (1 - t)**2 * t * P1[1] + 3 * (1 - t) * t**2 * P2[1] + t**3 * P3[1]
    
    return Bx, By

def third_order_bezier_curve(start_x, start_y, control_1_x, control_1_y, control_2_x, control_2_y, end_x, end_y):
        """
        2개의 제어점을 가지는 3차 베지어 곡선을 계산
        
        Parameters:
        - start_x, start_y: 시작점 좌표
        - control_1_x, control_1_y: 첫 번째 제어점 좌표
        - control_2_x, control_2_y: 두 번째 제어점 좌표
        - end_x, end_y: 끝점 좌표
        
        Returns:
        - x_vals, y_vals: 베지어 곡선 위의 좌표 리스트 (x, y)
        """

        num_points = num_points=40.0/0.5 
        t_vals = np.linspace(0, 1, int(num_points))
        x_vals = []
        y_vals = []

        for t in t_vals:
            x = (1 - t)**3 * start_x + 3 * (1 - t)**2 * t * control_1_x + 3 * (1 - t) * t**2 * control_2_x + t**3 * end_x
            y = (1 - t)**3 * start_y + 3 * (1 - t)**2 * t * control_1_y + 3 * (1 - t) * t**2 * control_2_y + t**3 * end_y
            x_vals.append(x)
            y_vals.append(y)
        
        return x_vals, y_vals


def update_vehicle_position(x, y, z, angle, speed, dt):
    # 차량이 1 타임스텝 동안 이동한 후의 위치를 계산
    new_x = x + speed * np.cos(angle) * dt
    new_y = y  # 차량은 기본적으로 y는 변경되지 않음 (차선 변경시만 y 변경)
    new_z = z  # z는 변화하지 않음 (2D 경로)
    return new_x, new_y, new_z

    
def sat_collision_check(object1_corners, object2_corners):
    def project_to_axis(corners, axis):
        projections = np.dot(corners, axis)
        return np.min(projections), np.max(projections)

    def get_axes(corners):
        axes = []
        for i in range(len(corners)):
            # 벡터 (A[i] -> A[i+1])에서 수직 벡터를 구하여 축을 만듬
            edge = corners[(i + 1) % len(corners)] - corners[i]
            normal = np.array([-edge[1], edge[0]])  # 90도 회전
            normal = normal / np.linalg.norm(normal)  # 단위 벡터로 정규화
            axes.append(normal)
        return axes

    object1_axes = get_axes(object1_corners)
    object2_axes = get_axes(object2_corners)

    for axis in object1_axes + object2_axes:
        min1, max1 = project_to_axis(object1_corners, axis)
        min2, max2 = project_to_axis(object2_corners, axis)
        
        # 만약 두 사각형이 분리되어 있으면 충돌이 아님
        if max1 < min2 or max2 < min1:
            return False
    return True

