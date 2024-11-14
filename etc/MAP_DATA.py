import json
import os

class Lane:
    def __init__(self):
        self.lane_id = 0
        self.laneposition = ""
        self.MainID = ""
        self.RightID = ""
        self.LeftID = ""
        self.Is_joining_lane = False
        self.Is_expansion_lane = False
        self.total_length = 0.0
        self.road_id = 0
        self.ToID = []
        self.FromID = []
        self.lane_x_list = []
        self.lane_y_list = []
        self.point_list = []
        self.distance = []
        self.angle = []
        self.d = []

class Road:
    def __init__(self):
        self.road_id = 0
        self.road_length = 0.0
        self.lane_order = []
        self.lane_id = []

def organize_lanes(map_data):
    lane_coordinates = {
        1: {"x": [], "y": []},
        2: {"x": [], "y": []},
        3: {"x": [], "y": []},
        4: {"x": [], "y": []}
    }
    target_roads = [1, 10, 9, 2, 3, 4, 5, 6, 7, 8]

    for road_id in target_roads:
        for lane_id, lane_obj in map_data.lane_dic.items():
            if lane_obj.road_id == road_id:  # 현재 도로가 목표 도로인지 확인
                lane_number = lane_obj.lane_id  # lane_id가 차선 번호라고 가정
                if lane_number in lane_coordinates:
                    lane_coordinates[lane_number]["x"].extend(lane_obj.lane_x_list)
                    lane_coordinates[lane_number]["y"].extend(lane_obj.lane_y_list)

    return lane_coordinates


class MAP_DATA:
    def __init__(self):
        self.x = []
        self.y = []
        self.road_id_list = []
        self.lane_id_list = []
        self.joker_road_id_list = []
        self.joker_lap_x = []
        self.joker_lap_y = []
        self.joker_lane_id_list = []
        self.joker_distance_list = []
        self.normal_road_id_list = []
        self.normal_lap_x = []
        self.normal_lap_y = []
        self.normal_lane_id_list = []
        self.normal_frenet_d_list = []
        self.normal_distance_list = []
        self.distance_list = []
        self.lane_dic = {}
        self.lane_main_id_dic = {}
        self.str_lane_dic = {}
        self.road_dic = {}
        self.lane_id_to_road_id = {}

    def json_map_parser(self):
        data_file_path = os.path.join('map')
        map_data_detail_file = os.path.join(data_file_path, 'map_data_detail.json')
        road_relation_file = os.path.join(data_file_path, 'road_relation.json')

        try:
            with open(map_data_detail_file, 'r') as json_file:
                root = json.load(json_file)
        except Exception as e:
            print(f"Failed to open or parse JSON file: {map_data_detail_file}, Error: {e}")
            return

        try:
            with open(road_relation_file, 'r') as json_file2:
                root2 = json.load(json_file2)
        except Exception as e:
            print(f"Failed to open or parse JSON file: {road_relation_file}, Error: {e}")
            return

        map_name = root["name"]
        roads = root["roads"]

        for road in roads:
            road_id = road["road_id"]
            lanes = road["lanes"]
            for lane in lanes:
                lane_obj = Lane()
                lane_obj.lane_id = lane["ID"]
                lane_obj.laneposition = lane["laneposition"]
                lane_obj.MainID = lane["MainID"][0]
                lane_obj.RightID = lane["RightID"][0]
                lane_obj.LeftID = lane["LeftID"][0]
                lane_obj.Is_joining_lane = lane["Is_joining_lane"]
                lane_obj.Is_expansion_lane = lane["Is_expansion_lane"]
                lane_obj.total_length = lane["TotalLength"]
                lane_obj.road_id = road_id
                
                lane_obj.ToID = [to_id for to_id in lane["ToID"]]
                lane_obj.FromID = [from_id for from_id in lane["FromID"]]
                
                for i in range(len(lane["x"])):
                    self.x.append(lane["x"][i])
                    self.y.append(lane["y"][i])
                    self.road_id_list.append(road_id)
                    self.lane_id_list.append(lane_obj.lane_id)

                    # Joker and normal road handling
                    if road_id in [2, 10, 11, 14, 16, 17, 20, 29, 30, 31, 32]:
                        self.joker_road_id_list.append(road_id)
                        self.joker_lap_x.append(lane["x"][i])
                        self.joker_lap_y.append(lane["y"][i])
                        self.joker_lane_id_list.append(lane_obj.lane_id)
                        self.joker_distance_list.append(lane["Distance"][i])
                    elif road_id in [9, 12, 13, 15, 18, 19, 25, 28]:
                        self.normal_road_id_list.append(road_id)
                        self.normal_lap_x.append(lane["x"][i])
                        self.normal_lap_y.append(lane["y"][i])
                        self.normal_lane_id_list.append(lane_obj.lane_id)
                        self.normal_distance_list.append(lane["Distance"][i])
                    else:
                        # Handle non-normal and non-joker roads
                        self.joker_road_id_list.append(road_id)
                        self.joker_lap_x.append(lane["x"][i])
                        self.joker_lap_y.append(lane["y"][i])
                        self.joker_lane_id_list.append(lane_obj.lane_id)
                        self.joker_distance_list.append(lane["Distance"][i])
                        
                        self.normal_road_id_list.append(road_id)
                        self.normal_lap_x.append(lane["x"][i])
                        self.normal_lap_y.append(lane["y"][i])
                        self.normal_lane_id_list.append(lane_obj.lane_id)
                        self.normal_distance_list.append(lane["Distance"][i])

                    self.distance_list.append(lane["Distance"][i])

                    point = [lane["x"][i], lane["y"][i]]
                    lane_obj.lane_x_list.append(lane["x"][i])
                    lane_obj.lane_y_list.append(lane["y"][i])
                    lane_obj.point_list.append(point)
                    lane_obj.distance.append(lane["Distance"][i])
                    lane_obj.angle.append(lane["Angle"][i])

                self.lane_dic[lane_obj.lane_id] = lane_obj
                self.lane_main_id_dic[lane_obj.MainID] = lane_obj
                self.str_lane_dic[lane_obj.MainID] = lane_obj

        map_name2 = root2["name"]
        roads2 = root2["roads"]

        for road2 in roads2:
            road_obj = Road()
            road_obj.road_id = road2["road_id"]
            road_obj.road_length = road2["road_length"]
            for i in range(len(road2["lane"])):
                road_obj.lane_order.append(road2["lane"][i])
                road_obj.lane_id.append(road2["lane_id"][i])
            self.road_dic[road_obj.road_id] = road_obj
            for lane_id in road_obj.lane_id:
                self.lane_id_to_road_id[lane_id] = road_obj.road_id

        print("Data Loaded successfully")

def main():
    # Example usage
    map_data = MAP_DATA()
    map_data.json_map_parser()

if __name__ == '__main__':
    main()