import ray


@ray.remote
class VehicleList:
    def __init__(self):
        self.vehicle_list = {}

    def set(self, vehicle):
        self.vehicle_list[vehicle.id] = vehicle

    def get(self, id):
        return self.vehicle_list[id]
    
    def getVehicleList(self):
        return self.vehicle_list



