import numpy as np
import pandas as pd
from config_settings import MAP_WIDTH, MAP_HEIGHT
import os
import sys


# """ Import SUMO library """
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    # from traci.exceptions import TraCIException
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")


import sumolib
from sumolib import net
import traci

def load_demand():
    demand_list = []
    demand = {}
    edge_list = list(traci.edge.getIDList())
    for edgeID in edge_list:
        v_num = traci.edge.getLastStepVehicleNumber(edgeID)
        if v_num == 0:
            demand_list.append(traci.edge.getStreetName(edgeID))
    demand_list = set(demand_list)
    street2edge = Street2Edge(demand_list)
    for e_id in street2edge:
        edgexy = EdgeXY(e_id)
        current_time = traci.simulation.getTime()
        demand[e_id] = [edgexy, current_time]
    return demand

def Street2Edge(demand_list):
    '''convert street name to edge id'''
    street2edge = set()
    for street_name in demand_list:
        for edge in traci.edge.getIDList():
            if traci.edge.getStreetName(edge) == street_name:
                edge_id = edge
                street2edge.add(edge_id)
                break
    return list(street2edge)

def EdgeXY(edge_id):
    '''get a xy coordinate of edgeid''' 
    Net = sumolib.net.readNet("file:///home/mak36/Desktop/sumo_motov/seoul_scenario/new2.net.xml") # file path
    EdgeXY = None
    for edge in Net.getEdges():
        if edge.getID() == edge_id:
            edge_shape = edge.getShape()
            EdgeXY = [round((edge_shape[0][0] + edge_shape[1][0]) / 2), round((edge_shape[0][1] + edge_shape[1][1]) / 2)]
    return EdgeXY



class DemandLoader(object):

    def __init__(self, timestep=1800, amplification_factor=1.0):
        self.timestep = timestep
        self.amplification_factor = amplification_factor
        self.demand = np.load('/home/mak36/Desktop/sumo_motov/seoul_scenario/hourly_demand.npy', allow_pickle=True)
        self.current_time = None
        self.hourly_demand = []

#----------------------------------------------------------------------------------------#

    def load(self, t, horizon=2):
        x = self.update_hourly_demand(t - self.timestep)
        demand = []

        for _ in range(horizon + 1):
            if abs(x)  <= 0.5:
                d = self.__compute_demand(x, self.hourly_demand[0:2])
            elif 0.5  <  x and x  <= 1.5:
                d = self.__compute_demand(x - 1, self.hourly_demand[1:3])
            elif 1.5  <  x and x  <= 2.5:
                d = self.__compute_demand(x - 2, self.hourly_demand[2:4])
            else:
                raise NotImplementedError

            x += self.timestep / 3600.0
            demand.append(d)

        latest_demand = self.load_latest_demand(t - self.timestep, t)
        return demand[1:], demand[0] - latest_demand

    def __compute_demand(self, x, d):
        return ((d[1] - d[0]) * x + (d[0] + d[1]) / 2) / 3600.0 * self.timestep * self.amplification_factor

    def update_hourly_demand(self, t, max_hours=4):
        localtime = traci.simulation.getTime() 
        current_time = localtime // 3600
        if len(self.hourly_demand) == 0 or self.current_time != current_time:
            if current_time < 6:
                self.current_time = current_time 
                self.hourly_demand = [self.load_demand_profile(current_time) for i in range(max_hours)]

        x = (localtime*60 - 30) / 60.0
        return x

    @staticmethod
    def load_demand_profile(self, t): # get demand per time charge xy
        demand = []
        t_demand = list(self.demand[t -1])
        
        street2edge = DemandLoader.Street2Edge(t_demand)
       
        for e_id in street2edge:
            EdgeXY = DemandLoader.EdgeXY(e_id)
            demand.append(EdgeXY)
            
        M = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        for (x, y) in demand:  
            M[x, y] += 1  
        return M

    @staticmethod
    def load_latest_demand(t_start, t_end):
        query = """
            SELECT x, y, demand
            FROM demand_latest
            WHERE t > {t_start} and t  < = {t_end};
                """.format(t_start = t_start, t_end = t_end)
        demand = pd.read_sql(query, engine, index_col=["x", "y"]).demand
        M = np.zeros((MAP_WIDTH, MAP_HEIGHT))
        for (x, y), c in demand.iteritems():
            M[x, y] += c
        return M
