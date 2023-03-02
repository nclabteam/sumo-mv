# from Vehicle import Vehicle
# from vehicles import vehicle_list
# veh1 = Vehicle(1)
# veh2 = Vehicle(2)
# vehicle_list[1] = veh1
# vehicle_list[2] = veh2
# print(Vehicle.vehicles)
# print(vehicle_list)
# v1 = Vehicle.get(1)
# print(v1.id)
# import pickle

# with open('./vehicle_list.pkl', 'rb') as vl_file:
#     vehicle_list = pickle.load(vl_file)

# print(vehicle_list)
import numpy as np
import time
t = time.time()
hourofday = (t/3600) / 24.0 * 2 * np.pi #####
dayofweek = 1 / 7.0 * 2 * np.pi #####
print([np.sin(hourofday), np.cos(hourofday), np.sin(dayofweek), np.cos(dayofweek)])