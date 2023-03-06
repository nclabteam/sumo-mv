import numpy as np
from skimage.transform import downscale_local_mean, resize
import os
import sys
from common import status_codes, mesh
from demand_loader import DemandLoader
import ray

if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    # from traci.exceptions import TraCIException
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import net
import traci

MAP_WIDTH = 600
MAP_HEIGHT = 600
MAX_MOVE = 20
NUM_SUPPLY_DEMAND_MAPS = 5
DESTINATION_PROFILE_SPATIAL_AGGREGATION = 5
DESTINATION_PROFILE_TEMPORAL_AGGREGATION = 3
MIN_DISPATCH_CYCLE = 60 * 7.5
L = MAX_MOVE * 2 + 1

class FeatureConstructor(object):

    def __init__(self):
        self.demand_loader = DemandLoader()
        self.t = 0
        self.fingerprint = (100000, 0)
        self.reachable_map = self.load_reachable_map()
        self.state_space = [(x, y) for x in range(MAP_WIDTH) for y in range(MAP_HEIGHT) if self.reachable_map[x, y] == 1]
        self.DT = self.load_dt_map()
        self.n_diffusions = 3
        self.D_out, self.D_in = self.build_diffusion_filter()
        self.d_entropy = self.build_diffusion_entropy_map()
        self.TT = None
        self.OD = None
        self.supply_maps = None
        self.demand_maps = None


    def action_space_iter(self, x, y):
        yield (0, 0)
        for ax in range(-MAX_MOVE, MAX_MOVE + 1):
            for ay in range(-MAX_MOVE, MAX_MOVE + 1):
                if ax == 0 and ay == 0:
                    continue
                x_ = x + ax
                y_ = y + ay
                if self.is_reachable(x_, y_):
                    yield (ax, ay)

    def load_reachable_map(self):
        return np.load('/home/mak36/Desktop/sumo_motov/seoul_scenario/reachable_map.npy')

    def load_dt_map(self):
        return np.load('/home/mak36/Desktop/sumo_motov/seoul_scenario/tt_map.npy') / MIN_DISPATCH_CYCLE

    def build_diffusion_filter(self):
        D_out = np.exp(-(self.DT) ** 2 + 1) / (L ** 2)
        D_in = np.zeros((MAP_WIDTH, MAP_HEIGHT, L, L))
        for x, y in self.state_space:
            for ax, ay in self.action_space_iter(x, y):
                axi, ayi = MAX_MOVE + ax, MAX_MOVE + ay
                D_in[x, y, axi, ayi] = D_out[x + ax, y + ay, -axi-1, -ayi-1]
        return D_out, D_in

    def build_diffusion_entropy_map(self):
        entropy = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        for x, y in self.state_space:
            entropy[x, y] = -(self.D_out[x, y] * np.log(self.D_out[x, y] + 1e-6)).sum()
        entropy /= np.log(L ** 2 + 1e-6)
        diffused_entropy = [entropy] + self.diffusion_convolution(entropy, self.D_out, self.n_diffusions - 1)
        return diffused_entropy

    def update_time(self, current_time):
        self.t = traci.simulation.getTime()

    # TO DO Check the number of idle and occupied vehicles from global vehicles list
    def update_supply(self, duration=MIN_DISPATCH_CYCLE * 2): ##### need to fix vehicle state dictionary first.
        # vehicles = ray.get_actor("vehicles_list")
        vehicles_actor = ray.get_actor("vehicles_actor")
        vehicles = ray.get(vehicles_actor.getVehicleList.remote())
        # idle = vehicles[(vehicles.status == status_codes.IDLE)]
        idle = []
        occupied = []
        idle_coords = []
        occupied_cords = []
        vehicles_traci=traci.vehicle.getIDList()
        print(vehicles_traci)
        for veh in vehicles.values():
            veh_id = -1
            try:
                veh_id = vehicles_traci.index(veh.id)
            except ValueError:
                pass
            if veh.state == 'IDLE' and veh_id != -1:
                lon,lat = traci.vehicle.getPosition(vehicles_traci[veh_id])
                idle.append(veh)
                idle_coords.append((lon,lat))
            elif veh.state == "ASSIGNED" and veh_id != -1:
                occupied.append(veh)
                lon,lat = traci.vehicle.getPosition(vehicles_traci[veh_id])
                occupied_cords.append((lat,lon))
                
        # occupied = vehicles[vehicles.status == status_codes.ASSIGNED] #####
        # occupied = occupied[occupied.time_to_destination <= duration] #####
        print("++++++++ {}".format(idle_coords))
        idle_map = self.construct_supply_map(idle_coords) ##### get lat and longitude from traci done
        dropoff_map = self.construct_supply_map(occupied_cords) #####
        print("++++++++ idle : {} occupied : {}".format(len(idle),len(occupied)))
        self.supply_maps = [idle_map, dropoff_map] 
        self.diffused_supply = [] 
        for s in self.supply_maps: 
            self.diffused_supply += self.diffusion_convolution(s, self.D_in, self.n_diffusions) 

    def update_demand(self, t, demand_normalized_factor=0.1, horizon=2): 
        profile, diff = self.demand_loader.load(t, horizon=horizon) 
        self.demand_maps = [d * demand_normalized_factor for d in profile] + [diff] 
        self.diffused_demand = [] 
        for d in self.demand_maps: 
            self.diffused_demand += self.diffusion_convolution(d, self.D_out, self.n_diffusions)  
        
    def diffusion_convolution(self, img, d_filter, k):
        M = img
        diffused_maps = [] 
        for _ in range(k): 
            M = self.diffuse_map(M, d_filter) 
            diffused_maps.append(M) 
        return diffused_maps 

    def trip_diffusion_convolution(self, img, trip_filter):
        n = DESTINATION_PROFILE_SPATIAL_AGGREGATION
        M = downscale_local_mean(img, (n, n))
        M = np.tensordot(trip_filter, M)
        M = resize(M, img.shape, mode='edge')
        return M
    
    def diffuse_map(self, img, d_filter):
        padded_map = np.pad(img, MAX_MOVE, "constant")
        diffused_map = self.construct_initial_map()
        for x, y in self.state_space:
            diffused_map[x, y] = (padded_map[x : x + L, y : y + L] * d_filter[x, y]).sum()
        return diffused_map

    def update_fingerprint(self, fingerprint):
        self.fingerprint = fingerprint

    def construct_current_features(self, x, y):
        M = self.get_supply_demand_maps()
        t = self.get_current_time()
        f = self.get_current_fingerprint()
        l = (x, y)
        print("inside construct_current_feratures : {}".format(l))
        s, actions = self.construct_features(t, f, l, M)
        return s, actions

    def construct_features(self, t, f, l, M):
        print("inside construct_feratures : {}".format(l))
        state_feature = self.construct_state_feature(t, f, l, M)
        actions, action_features = self.construct_action_features(t, l, M)
        s = (state_feature, action_features)
        return s, actions

    def construct_state_feature(self, t, f, l, M):
        x, y = l
        print("inside construct_state_feratures : {}".format(l))
        state_feature = [m.mean() for m in M[:NUM_SUPPLY_DEMAND_MAPS]]
        state_feature += [m[x, y] for m in M]
        state_feature += [m[x, y] for m in self.d_entropy]
        state_feature += self.construct_time_features(t) + self.construct_fingerprint_features(f)
        state_feature += self.construct_fingerprint_features(f)
        return state_feature

    def construct_action_features(self, t, l, M):
        actions = []
        action_features = []

        for ax, ay in self.action_space_iter(*l):
            a = (ax, ay)
            feature = self.construct_action_feature(t, l, M, a)
            if feature is not None:
                actions.append(a)
                action_features.append(feature)

        return actions, action_features

    def is_reachable(self, x, y):
        return 0  <= x and x  <  MAP_WIDTH and 0  <= y and y  <  MAP_HEIGHT and self.reachable_map[x, y] == 1

    def construct_action_feature(self, t, l, M, a):
        x, y = l
        ax, ay = a
        x_ = x + ax
        y_ = y + ay
        tt = self.get_triptime(x, y, ax, ay)
        if tt  <= 1:
            action_feature = [m[x_, y_] for m in M]
            action_feature += [m[x_, y_] for m in self.d_entropy]
            action_feature += [tt]
            # action_feature += self.construct_location_features((x_, y_)) + [tt]
            return action_feature

        return None

    def get_triptime(self, x, y, ax, ay):
        return self.DT[x, y, ax + MAX_MOVE, ay + MAX_MOVE]

    def get_supply_demand_maps(self):
        t = traci.simulation.getTime()
        self.update_supply()
        self.update_demand(t)
        supply_demand_maps = self.supply_maps + self.demand_maps
        diffused_maps = self.diffused_supply + self.diffused_demand
        return supply_demand_maps + diffused_maps

    def construct_initial_map(self, w=MAP_WIDTH, h=MAP_HEIGHT):
        return np.zeros((w, h), dtype=np.float32)

    def construct_supply_map(self, locations):
        supply_map = self.construct_initial_map()
        for lon, lat in locations:
            print("longitude latitute {} {}".format(lon,lat))
            x, y = mesh.convert_lonlat_to_xy(lon, lat) #####
            print("x y {} {}".format(x,y))
            supply_map[x, y] += 1.0
        return supply_map
    # TO DO convert severn days to one day only pending
    def construct_time_features(self, timestamp):
        t = traci.simulation.getTime() #####
        hourofday = (t/3600) / 24.0 * 2 * np.pi #####
        dayofweek = 1 / 7.0 * 2 * np.pi #####
        return [np.sin(hourofday), np.cos(hourofday), np.sin(dayofweek), np.cos(dayofweek)]
    
    def construct_fingerprint_features(self, fingerprint):
        iteration, epsilon = fingerprint
        return [np.log(1 + iteration / 60.0), epsilon]

    def construct_location_features(self, l):
        x, y = l
        x_norm = float(x - MAP_WIDTH / 2.0) / MAP_WIDTH * 2.0
        y_norm = float(y - MAP_HEIGHT / 2.0) / MAP_HEIGHT * 2.0
        return [x_norm, y_norm]

    def get_current_time(self):
        t = self.t
        return t

    def get_current_fingerprint(self):
        f = self.fingerprint
        return f