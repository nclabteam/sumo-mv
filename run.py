# import sys
import os
from copy import deepcopy
import logging
import pathlib
from pprint import pformat
import pickle
#ray
import gym
import ray
from ray import tune
import tensorflow as tf
import argparse
from ray.rllib.agents.dqn import dqn
import marlenvironment
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.models import ModelV2, ModelCatalog
from ray.rllib.agents.dqn.dqn_tf_policy import DQNTFPolicy
from feature_constructor import FeatureConstructor

#from ray.rllib.utils.typing import ModelConfigDict
from ray.rllib.models.preprocessors import Preprocessor
from Vehicle import Vehicle
# sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from vehicles import VehicleList

from dqn_settings import NUM_SUPPLY_DEMAND_HISTORY, FLAGS, INITIAL_EPSILON, FINAL_EPSILON, MAX_MEMORY_SIZE, TARGET_UPDATE_INTERVAL
from config_settings import TIMESTEP, DEFAULT_LOG_DIR, MAP_WIDTH, MAP_HEIGHT, ENTERING_TIME_BUFFER


parser = argparse.ArgumentParser()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dqntrain")

parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Trainer state.")

class MyPreprocessorClass(Preprocessor):
    def __init__(self, *arg, **kwargs):
        super().__init__(*arg, **kwargs)
        self.feature_constructor = FeatureConstructor()
       
    def transform(self, observation):
        s = self.feature_constructor.construct_current_features(observation[0],observation[1])
        s_feature, a_features = s
        sa_input = np.array([s_feature + a_feature for a_feature in a_features], dtype=np.float32)
        return sa_input
   
    def _init_shape(self, obs_space: gym.Space, options: dict):
        return (obs_space.shape[0], )
    

class MyDQNModel(ModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(MyDQNModel, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        sa_input = tf.keras.layers.Input(shape=(58, ), dtype='float32')
        x = tf.keras.layers.Dense(100, activation='relu', name='dense_1')(sa_input)
        x = tf.keras.layers.Dense(100, activation='relu', name='dense_2')(x)
        q_value = tf.keras.layers.Dense(1, name='q_value')(x)
        
        self.base_model = tf.keras.Model(inputs=sa_input, outputs=q_value)
    
    def forward(self, input_dict, state):
        s = input_dict['obs']
        s_feature, a_features = s
        q = self.base_model(s_feature + a_features).numpy()[:, 0]
        return q, state
    
class MyDQNPolicy(DQNTFPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config["optimizer"] = {
            "type": "rmsprop",
            "lr": 0.00025,
            "momentum": 0.95,
            
        }
                
    def build_model(self):
        self.q_model = ModelCatalog.get_model_v2(self.observation_space,
                                                 self.action_space,
                                                 self.config["num_outputs"],
                                                 model_config=self.config["model"],
                                                 name="q_model")
        self.target_q_model = ModelCatalog.get_model_v2(self.observation_space,
                                                        self.action_space,
                                                        self.config["num_outputs"],
                                                        model_config=self.config["model"],
                                                        name="target_q_model")
        
    def compute_action(self, observation, **kwargs):
        # First calculate the available actions
        available_actions = self.get_available_actions(observation)
        
        # Apply epsilon-greedy exploration strategy
        if np.random.rand() < self.config["exploration_config"]["epsilon"]:
            action = np.random.choice(available_actions)
        else:
            q_values = self.q_network.get_q_values(observation)
            action = np.argmax(q_values[available_actions])
            
        return action

    def learn_on_batch(self, samples):
        pass

    
if __name__ == '__main__':
    args = parser.parse_args()
    ray.init()
    tune.register_env("sumo_test_env", marlenvironment.env_creator)

    vehicles_actor = VehicleList.options(name="vehicles_actor").remote()
    logger.info("++++++++++ ray initilized ++++++++++++++++++++++++++++++++++++++++++")
    # Algorithm.
    policy_class = MyDQNPolicy
    config = dqn.DEFAULT_CONFIG
    config["exploration_config"]["initial_epsilon"] = 1.0
    config["exploration_config"]["final_epsilon"] = 0.01
    config["exploration_config"]["epsilon_timesteps"] = 3000
    config["target_network_update_freq"] = 50
    config["replay_buffer_config"]["capacity"] = 1000000
    config["framework"] = "tf"
    config["gamma"] = 0.98
    config["min_iter_time_s"] = 5
    config["num_gpus"] = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
    config["num_workers"] = FLAGS.num_workers
    config["rollout_fragment_length"] = 200
    config["train_batch_size"] = 128
    
    # Load default Scenario configuration for the LEARNING ENVIRONMENT
    scenario_config = deepcopy(marlenvironment.DEFAULT_SCENARIO_CONFIG)
    scenario_config["seed"] = 42
    scenario_config["log_level"] = "INFO"
    scenario_config["sumo_config"]["sumo_connector"] = 'traci'
    scenario_config["sumo_config"]["sumo_gui"] = 'sumo-gui'
    filename = "{}/seoul_scenario/osm.sumocfg".format(
          pathlib.Path(__file__).parent.absolute())
    scenario_config["sumo_config"]["sumo_cfg"] = filename
    scenario_config["sumo_config"]["sumo_params"] = [
        "--collision.action", "warn"
    ]
    scenario_config["sumo_config"]["trace_file"] = True
    scenario_config["sumo_config"]["end_of_sim"] = 28800  # [s]
    scenario_config["sumo_config"]["learning_starts"] = 3600
    scenario_config["sumo_config"]["update_freq"] = 600  # number of traci.simulationStep()
    # for each learning step.
    scenario_config["sumo_config"]["log_level"] = "INFO"
    logger.info("Scenario Configuration: \n %s", pformat(scenario_config))
    # Associate the agents with their configuration.
    agent_dict = {}
    vehicles = {}

    for i in range(5):
        veh_id = 'agent' + str(i)
        v_id = veh_id
        vagent = Vehicle(veh_id)
        vehicles[v_id] = vagent
        vehicles_actor = ray.get_actor("vehicles_actor")
        vehicles_actor.set.remote(vagent)
        agent_dict[v_id] = deepcopy(marlenvironment.DEFAULT_AGENT_CONFIG)
    agent_init = agent_dict
    logger.info("Agents Configuration: \n %s", pformat(agent_init))
    # with open('vehicle_list.pkl', 'wb') as file:
    #     pickle.dump(vehicles, file)
    # Register your custom fn
    ModelCatalog.register_custom_model("my_dqn", MyDQNModel)
    ModelCatalog.register_custom_preprocessor("my_prep", MyPreprocessorClass)
    
    # MARL Environment Init
    env_config = {
        "agent_init": agent_init,
        "scenario_config": scenario_config,
    }
    marl_env = marlenvironment.SUMOTestMultiAgentEnv(env_config)
    config['model']['custom_preprocessor'] = 'my_prep' # Set custom_preprocessor.

    # Config for the DQN trainer from the MARLEnv
    policies = {}
    for agent in marl_env.get_agents():
        agent_policy_params = {}
        policies[agent] = (policy_class, marl_env.get_obs_space(),
                           marl_env.get_action_space(agent),
                           agent_policy_params)
    config["multiagent"]["policies"] = policies
    config["multiagent"]["policy_mapping_fn"] = lambda agent_id, episode, **kwargs: agent_id
    config["multiagent"]["policies_to_train"] = ["dqn_policy"]

    config["env"] = "sumo_test_env"
    config["env_config"] = env_config
    logger.info("DQNConfiguration: \n %s", pformat(config))

    stop = {
        "timesteps_total": 100,
    }

    exploration_config = {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.01,
        "epsilon_timesteps": 3000,
    }

    # Run the experiment.
    results = tune.run(
        "DQN",
        config=config,
        stop=stop,
        verbose=1,
        checkpoint_freq=10,
        restore=args.from_checkpoint)

    # And check the results.
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)

ray.shutdown()