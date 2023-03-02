import numpy as np
import os
import sys

# """ Import SUMO library """
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    # from traci.exceptions import TraCIException
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

import traci


'''
def convert_lonlat_to_xy(lon, lat):
    x = (lon - MIN_LON) / DELTA_LON
    x = int(min(max(x, 0), MAP_WIDTH - 1))
    y = (lat - MIN_LAT) / DELTA_LAT
    y = int(min(max(y, 0), MAP_HEIGHT - 1))
    return x, y
'''

def convert_xy_to_lonlat(x, y):
    lon, lat = traci.simulation.convertGeo(x,y,fromGeo=False)
    return lon, lat

def convert_lonlat_to_xy(lon, lat):
    x, y = traci.simulation.convertGeo(lon, lat, fromGeo=True)
    return x, y
'''
def lon2X(lons):
    X = np.int32(np.minimum(np.maximum((lons - MIN_LON) / DELTA_LON, 0), MAP_WIDTH - 1))
    return X

def lat2Y(lats):
    Y = np.int32(np.minimum(np.maximum((lats - MIN_LAT) / DELTA_LAT, 0), MAP_HEIGHT - 1))
    return Y

def X2lon(X):
    lons = MIN_LON + DELTA_LON * (np.minimum(np.maximum(X, 0), MAP_WIDTH - 1) + 0.5)
    return lons

def Y2lat(Y):
    lats = MIN_LAT + DELTA_LAT * (np.minimum(np.maximum(Y, 0), MAP_HEIGHT - 1) + 0.5)
    return lats
'''