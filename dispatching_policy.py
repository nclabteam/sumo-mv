import numpy as np
from collections import defaultdict
import Vehicle
import traci  # noqa
import ray
import pickle
MAP_WIDTH = 600
MAP_HEIGHT = 600

class DispatchingPolicy(object):
    def __init__(self, reject_distance=500):
        self.reject_distance = reject_distance  # meters
        self.reject_wait_time = 120         # seconds
        self.unit_length = 1                   # mesh size in meters
        self.max_locations = 40                 # max number of origin/destination points

    def find_available_vehicles(self, vehicles):
        '''get list of idle vehicles'''
        # vehicles_list = ray.get_actor("vehicles_list")
        # with open('/home/mak36/Desktop/sumo_motov/vehicle_list.pkl', 'rb') as vl_file:
        #     vehicle_list = pickle.load(vl_file)
        idle_vehicles = []
        for id in vehicles:
            vehicles_actor = ray.get_actor("vehicles_actor")
            vagent = ray.get(vehicles_actor.get.remote(id))
            # state = vehicle_list[id].state
            state = vagent.state
            if state == 'IDLE':
                idle_vehicles.append(id)
        return idle_vehicles

    def create_command(self, vehicle_id, edge_id, T):
        command = {}
        command["vehicle_id"] = vehicle_id
        command["edge_id"] = edge_id
        command["current_time"] = T
        return command

    def assign_nearest_vehicle(self, rid, request_xy, vehicles, reject_range, T):
        '''compute a distance and get a nearest vehicle'''
        assignments = []
        min_d = 1000
        assigned_vid = None
        for vid in vehicles:
            v_x, v_y = traci.vehicle.getPosition(vid)
            d = traci.simulation.getDistance2D(v_x, v_y, request_xy[0], request_xy[1], isGeo=False, isDriving=True)
            if (d < min_d) and (d < reject_range) and (d > 0):
                min_d = d
                assigned_vid = vid
        if assigned_vid == None:
            return assignments
        assignments.append((assigned_vid, rid, T))
        return assignments

    # traci.simulation.getDistance2D(x, y, -150, 97, isGeo=False, isDriving=True) 
    def dispatch(self, vehicles, requests):
        '''requests -> dictionary 'edgeid' : [(x,y), time]'''
        commands = []
        vehicle_ids = []
        for v in vehicles:
            vehicle_ids.append(v)
                        
        vehicle_ids = self.find_available_vehicles(vehicle_ids) # Idle 한 vehicle 들 가져오기
        n_vehicles = len(vehicle_ids)
        if n_vehicles == 0:
            return commands
        
        reject_range = int(self.reject_distance)
        for target_rid in requests.keys():
            assignments = self.assign_nearest_vehicle(target_rid, requests[target_rid][0], vehicle_ids, reject_range, requests[target_rid][1])
            if len(assignments) == 0:
                for vid, rid, T in assignments:
                    commands.append(self.create_command(vid, rid, T))
                    vehicle_ids.remove(vid)
        return commands