""" Example MARL Environment for RLLIB SUMO Utlis

    Author: Lara CODECA lara.codeca@gmail.com

    See:
        https://github.com/lcodeca/rllibsumoutils
        https://github.com/lcodeca/rllibsumodocker
    for further details.
"""

import collections
import logging
import os
import sys
from pprint import pformat
import random
from numpy.random import RandomState
from Vehicle import Vehicle
from dispatching_policy import DispatchingPolicy
import numpy as np
import demand_loader
from demand_loader import DemandLoader
import pickle
import ray
import gym
from ray.rllib.env import MultiAgentEnv

from utils import SUMOUtils, sumo_default_config

# """ Import SUMO library """
if "SUMO_HOME" in os.environ:
    sys.path.append(os.path.join(os.environ["SUMO_HOME"], "tools"))
    # from traci.exceptions import TraCIException
    import traci.constants as tc
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

###############################################################################

logger = logging.getLogger(__name__)

###############################################################################


def env_creator(config):
    """Environment creator used in the environment registration."""
    logger.info("Environment creation: SUMOTestMultiAgentEnv")
    return SUMOTestMultiAgentEnv(config)


###############################################################################

MS_TO_KMH = 3.6
MAX_MOVE = 20

class SUMOSimulationWrapper(SUMOUtils):
    """A wrapper for the interaction with the SUMO simulation"""

    def _initialize_simulation(self):
        """Specific simulation initialization."""
        try:
            super()._initialize_simulation()
            '''
            edge_list = list(self.traci_handler.edge.getIDList())
            veh_id = 'v'
            for i in range(30):
                route_id = str(i)
                veh_id = 'v' + str(i)
                route_list = random.sample(edge_list, 1)
                self.traci_handler.route.add(route_id, [route_list[0]])
                self.traci_handler.vehicle.add(veh_id, route_id)
            '''
        except NotImplementedError:
            pass

    def _initialize_metrics(self):
        """Specific metrics initialization"""
        try:
            super()._initialize_metrics()
        except NotImplementedError:
            pass
        self.veh_subscriptions = dict()
        self.collisions = collections.defaultdict(int)
        self.dispatching_policy = DispatchingPolicy()

    def _default_step_action(self, agents):
        """Specific code to be executed in every simulation step"""
        try:
            super()._default_step_action(agents)
        except NotImplementedError:
            pass
        # get collisions
        collisions = self.traci_handler.simulation.getCollidingVehiclesIDList()
        logger.debug("Collisions: %s", pformat(collisions))
        for veh in collisions:
            self.collisions[veh] += 1
        # get subscriptions
        self.veh_subscriptions = self.traci_handler.vehicle.getAllSubscriptionResults()
        for veh, vals in self.veh_subscriptions.items():
            logger.debug("Subs: %s, %s", pformat(veh), pformat(vals))
        running = set()
        for agent in agents:
            if agent in self.veh_subscriptions:
                running.add(agent)
        if len(running) == 0:
            logger.info("All the agent left the simulation..")
            self.end_simulation()
        
        # vehicles disappear when they reach destination (route[-1])
        # before disapear, set a new destination to vehicles
        edge_list = list(self.traci_handler.edge.getIDList())
        veh_list = self.traci_handler.vehicle.getIDList()
        agent_veh_list =[]
        simtime = self.traci_handler.simulation.getTime()
        # with open('/home/mak36/Desktop/sumo_motov/vehicle_list.pkl', 'rb') as vl_file:
        #     vehicle_list = pickle.load(vl_file)
        # in your envs
        for vid in veh_list:
            route = list(self.traci_handler.vehicle.getRoute(vid))
            if self.traci_handler.vehicle.getRoadID(vid) == route[-1]:
                if 'v' in vid:
                    edge = random.choice(edge_list)
                    self.traci_handler.vehicle.changeTarget(vid, edge)
                else:
                    agent_veh_list.append(vid)
                    # if assigned vehicle (dispatched vehicle) reach its destination, then change the route and state 
                    edge = random.choice(edge_list)
                    self.traci_handler.vehicle.changeTarget(vid, edge)
                    # vagent = Vehicle.get(vid) #TO DO access vehicle object here. done this part
                    vehicles_actor = ray.get_actor("vehicles_actor")
                    vagent = ray.get(vehicles_actor.get.remote(vid))
                    # vagent = vehicle_list[vid]
                    vagent.set_idle(simtime)
                    # vehicle_list[vid] = vagent
                    vehicles_actor.set.remote(vagent)
                    print("Vagent id : {} State : {} Dest : {}".format(vagent.id,vagent.state,vagent.dest))
                    # with open('vehicle_list.pkl', 'wb') as file:
                    #     pickle.dump(vehicle_list, file)
                    
        
        
        # every 3600s (1 hour) run dispatching
        if simtime % 3600 == 0:
            print("inside simtime ")
            requests = demand_loader.load_demand() 
            print("demand loaded")
            # agent_vehicle = list(veh_list)
            agent_vehicle = list(agent_veh_list)
            dispatching_commands = self.dispatching_policy.dispatch(agent_vehicle, requests)
            for command in dispatching_commands:
                request_edgeid = command['edge_id']
                vid = command['vehicle_id']
                self.traci_handler.vehicle.changeTarget(vid, request_edgeid)
                # vehicle = vehicle_list[vid]
                print("======== changing state of {} to {}".format(vehicle.id,'assigned'))
                # vehicle.set_assigned(request_edgeid, command['current_time'])
                # vehicle_list[vid] = vehicle
                # with open('vehicle_list.pkl', 'wb') as file:
                #         pickle.dump(vehicle_list, file)
                vehicles_actor = ray.get_actor("vehicles_actor")
                print("======== changing state of {} to {}".format(vehicle.id,'assigned'))
                vehicle = ray.get(vehicles_actor.get.remote(vid))
                vehicle.set_assigned(request_edgeid, command['current_time'])
                # vehicle.set_state(vid, 'ASSIGNED', command['current_time']) ###################
                vehicles_actor.set.remote(vehicle)
            logger.info("---------디스패칭된 차량 숫자, idle 한 차량 숫자--------")
                  
        return True


###############################################################################


class SUMOAgent:
    """Agent implementation."""

    def __init__(self, agent, config):
        self.agent_id = agent
        self.config = config
        self.action_to_meaning = config["actions"]
        #self.action_to_meaning = dict()
        #for action in config["actions"]:
        #    self.action_to_meaning[action] = config["actions"]
        logger.debug(
            "Agent '%s' configuration \n %s", self.agent_id, pformat(self.config)
        )

    def step(self, action, sumo_handler): ############action
        """Implements the logic of each specific action passed as input."""
        logger.debug("Agent %s: action %d", self.agent_id, action)
        logger.debug(
            "Subscriptions: %s", pformat(sumo_handler.veh_subscriptions[self.agent_id])
        )
        #previous_speed = sumo_handler.veh_subscriptions[self.agent_id][tc.VAR_SPEED]
        #new_speed = previous_speed + self.action_to_meaning[action]
        #logger.debug("Before %.2f", previous_speed)
        #sumo_handler.traci_handler.vehicle.setSpeed(self.agent_id, new_speed)
        #logger.debug("After %.2f", new_speed)
        action = [random.randint(-MAX_MOVE, MAX_MOVE), random.randint(-MAX_MOVE, MAX_MOVE)]
        x, y = action[0], action[1]
        Edge = sumo_handler.traci_handler.simulation.convertRoad(x, y)
        Edge_id = Edge[0]
        sumo_handler.traci_handler.vehicle.changeTarget(self.agent_id, Edge_id)
            
        return

    def reset(self, sumo_handler): #tnwjd
        """Resets the agent and return the observation.""" ''
        edge_list = sumo_handler.traci_handler.edge.getIDList()
        try:
            route_id = self.agent_id
            route_list = random.sample(edge_list, 1)
            sumo_handler.traci_handler.route.add(route_id, route_list)
        except sumo_handler.traci_handler.exceptions.TraCIException as e:
            if "Could not add route" in str(e):
                pass
        # https://sumo.dlr.de/pyhandler.route.add(route, ["road"])
        # insert the agent in the simulation
        # traci.vehicle.adddoc/traci._route.html#RouteDomain-add
        # sumo_handler.traci_(self, vehID, routeID, typeID="DEFAULT_VEHTYPE",
        #   depart=None, departLane="first", departPos="base", departSpeed="0",
        #   arrivalLane="current", arrivalPos="max", arrivalSpeed="current",
        #   fromTaz="", toTaz="", line="", personCapacity=0, personNumber=0)
        sumo_handler.traci_handler.vehicle.add(self.agent_id, route_id)
        sumo_handler.traci_handler.vehicle.setColor(self.agent_id, (255,0,0))
        sumo_handler.traci_handler.vehicle.subscribeLeader(self.agent_id)
        sumo_handler.traci_handler.vehicle.subscribe(
            self.agent_id, varIDs=[tc.VAR_POSITION]
        )
        logger.info("Agent %s reset done.", self.agent_id)
        return self.agent_id, self.config["start"]


###############################################################################

DEFAULT_SCENARIO_CONFIG = {
    "sumo_config": sumo_default_config(),
    "agent_rnd_order": True,
    "log_level": "WARN",
    "seed": 42,
    #"misc": {
    #    "max_distance": 1000,  # [m]
    #},
}

DEFAULT_AGENT_CONFIG = { #################################33
    "origin": "road",
    "destination": "road",
    "start": 0,
    "max_speed": 60,
    "actions" : [random.randint(-MAX_MOVE, MAX_MOVE), random.randint(-MAX_MOVE, MAX_MOVE)],
}

######################################################################################
class SUMOTestMultiAgentEnv(MultiAgentEnv):
    """
    A RLLIB environment for testing MARL environments with SUMO simulations.
    """

    def __init__(self, config):
        """Initialize the environment."""
        super(SUMOTestMultiAgentEnv, self).__init__()

        self._config = config

        # logging
        level = logging.getLevelName(config["scenario_config"]["log_level"])
        logger.setLevel(level)

        # SUMO Connector
        self.simulation = None

        # Random number generator
        self.rndgen = RandomState(config["scenario_config"]["seed"])

        # Agent initialization
        self.agents_init_list = dict()
        self.agents = dict()
        for agent, agent_config in self._config["agent_init"].items():
            self.agents[agent] = SUMOAgent(agent, agent_config)

        # Environment initialization
        self.resetted = True
        self.episodes = 0
        self.steps = 0

    def seed(self, seed):
        """Set the seed of a possible random number generator."""
        self.rndgen = RandomState(seed)

    def get_agents(self):
        """Returns a list of the agents."""
        return self.agents.keys()

    def __del__(self):
        logger.info("Environment destruction: SUMOTestMultiAgentEnv")
        if self.simulation:
            del self.simulation

    ###########################################################################
    # OBSERVATIONS

    def get_observation(self, agent): # 차량 위치
        """
        Returns the observation of a given agent.
        See http://sumo.sourceforge.net/pydoc/traci._simulation.html
        """
        if agent in self.simulation.veh_subscriptions:
            position = (self.simulation.veh_subscriptions[agent][tc.VAR_POSITION])
            x, y = round(position[0]), round(position[1])
            ret = [x,y]
        else:
            ret =[0,0]
        logger.debug("Agent %s --> Obs: %s", agent, pformat(ret))
        return ret

    def compute_observations(self, agents): # 그대로
        """For each agent in the list, return the observation."""
        obs = dict()
        for agent in agents:
            obs[agent] = self.get_observation(agent)
        return obs

    ###########################################################################
    # REWARDS

    def get_reward(self, agent): # 매칭 타임 + 일한 시간으로 가자
        """Return the reward for a given agent."""
        speed = self.agents[agent].config[
            "max_speed"
        ]  # if the agent is not in the subscriptions
        # and this function is called, the agent has
        # reached the end of the road
        if agent in self.simulation.veh_subscriptions:
            speed = round(
                self.simulation.veh_subscriptions[agent][tc.VAR_SPEED] * MS_TO_KMH
            )
        logger.debug("Agent %s --> Reward %d", agent, speed)
                     
        return speed

    def compute_rewards(self, agents):
        """For each agent in the list, return the rewards."""
        rew = dict()
        for agent in agents:
            rew[agent] = self.get_reward(agent)
        return rew

    ###########################################################################
    # REST & LEARNING STEP

    def reset(self): 
        """Resets the env and returns observations from ready agents."""
        self.resetted = True
        self.episodes += 1
        self.steps = 0
        logger.info("#############env reset###########.")

        # Reset the SUMO simulation
        if self.simulation:
            del self.simulation

        self.simulation = SUMOSimulationWrapper(
            self._config["scenario_config"]["sumo_config"]
        )

        # Reset the agents
        waiting_agents = list()
        for agent in self.agents.values():
            agent_id, start = agent.reset(self.simulation)
            waiting_agents.append((start, agent_id))
        waiting_agents.sort()

        # Move the simulation forward
        starting_time = waiting_agents[0][0]
        self.simulation.fast_forward(starting_time)
        self.simulation._default_step_action(
            self.agents.keys()
        )  # hack to retrieve the subs

        # Observations
        initial_obs = self.compute_observations(self.agents.keys())

        return initial_obs

    def step(self, action_dict):
        """
        Returns observations from ready agents.

        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.

        Returns
        -------
            obs: New observations for each ready agent.
            rewards: Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones: Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos: Optional info values for each agent id.
        """
        self.resetted = False
        self.steps += 1
        logger.debug(
            "====> [SUMOTestMultiAgentEnv:step] Episode: %d - Step: %d <====",
            self.episodes,
            self.steps,
        )
        dones = {}
        dones["__all__"] = False

        shuffled_agents = sorted(
            action_dict.keys()
        )  # it may seem not smar to sort something that
        # may need to be shuffled afterwards, but it
        # is a matter of consistency instead of using
        # whatever insertion order was used in the dict
        if self._config["scenario_config"]["agent_rnd_order"]:
            # randomize the agent order to minimize SUMO's
            # insertion queues impact
            logger.debug("Shuffling the order of the agents.")
            self.rndgen.shuffle(shuffled_agents)  # in-place shuffle

        # Take action
        for agent in shuffled_agents:
            self.agents[agent].step(action_dict[agent], self.simulation)

        logger.debug("Before SUMO")
        ongoing_simulation = self.simulation.step(
            until_end=False, agents=set(action_dict.keys())
        )
        logger.debug("After SUMO")

        # end of the episode
        if not ongoing_simulation:
            logger.info("Reached the end of the SUMO simulation.")
            dones["__all__"] = True

        obs, rewards, infos = {}, {}, {}
        dones[agent] = agent not in self.simulation.veh_subscriptions
        obs[agent] = self.get_observation(agent)
        rewards[agent] = self.get_reward(agent)
        # infos[agent] = ""

        logger.debug("Observations: %s", pformat(obs))
        logger.debug("Rewards: %s", pformat(rewards))
        logger.debug("Dones: %s", pformat(dones))
        logger.debug("Info: %s", pformat(infos))
        logger.debug("========================================================")
        return obs, rewards, dones, infos

    ###########################################################################
    # ACTIONS & OBSERATIONS SPACE

    def get_action_space_size(self, agent):
        """Returns the size of the action space."""
        return len(self.agents[agent].config["actions"])

    def get_action_space(self, agent):
        """Returns the action space."""
        return gym.spaces.Discrete(self.get_action_space_size(agent))

    def get_set_of_actions(self, agent):
        """Returns the set of possible actions for an agent."""
        return set(range(self.get_action_space_size(agent)))

    def get_obs_space_size(self, agent):
        """Returns the size of the observation space."""
        return (self.agents[agent].config["max_speed"] + 1) * (
            self._config["scenario_config"]["misc"]["max_distance"] + 1
        )

    def get_obs_space(self):
        """Returns the observation space."""
        return gym.spaces.Box(-600, 600, shape=(2,), dtype=np.int16)
        
        '''
        return gym.spaces.MultiDiscrete(
            [                
                self.agents[agent].config["max_speed"] + 1,
                self._config["scenario_config"]["misc"]["max_distance"] + 1,
            ]
        )
        '''    
        
        
