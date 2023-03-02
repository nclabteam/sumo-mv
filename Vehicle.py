import pandas as pd

class Vehicle(object):
    vehicles = {}
        
    def __init__(self, vehicle_id):
        self.id = vehicle_id
        self.vehicle_id = vehicle_id
        self.state = 'IDLE'
        self.dest = None
        self.start_time = 0
        self.working_time = 0
        self.dispatch_count = 0
        Vehicle.vehicles[vehicle_id] = self
        
    def set_assigned(self, destination, t):
        self.state = 'ASSIGNED'
        self.dest = destination
        self.start_time = t
        Vehicle.vehicles[self.vehicle_id] = self
                
    def set_idle(self,t):
        self.state = 'IDLE'
        self.working_time = self.working_time + (t - self.start_time)
        self.dispatch_count += 1
        self.dest = None
        Vehicle.vehicles[self.vehicle_id] = self

    
        
    @classmethod
    def get(cls, vehicle_id):
        return cls.vehicles.get(vehicle_id, None)
            
