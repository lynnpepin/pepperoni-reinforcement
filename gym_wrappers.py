# pep-action-space
# pep-observation-space
# BHDEnv

from pepperoni import BridgeHoleDesign
from reinfrocement_utils import _normalize_01, _normalize_angle
import numpy as np
import gym

def observe_bridge_update(data, length = 20.0, width = 10.0, allowable_stress=200.0):
    """Preprocess data and provide as an observation.
    
    Return preprocessed observation dict.
    """
    raise NotImplementedError

def data_to_space(data):
    """Return a gym.spaces.Space() representation of a data.
    
    Data must be represntable as a gym.spaces.Space() instance.
    
    Recursive! caliborn_clasped_hands_wonder.jpg
    """


# TODO - Implement the above
# TODO - Barebones space tests

class BHDEnv(gym.Env):
    """
    BridgeHoleDesign environment.
    
    
    """
    
    def __init__(self, bridge = None,
                       length = 20.0,
                       height = 10.0,
                       allowable_stress = 200.0):
        
        self.__version__ = "0.1.0"
        
        # Set up bridge
        self.bridge = bridge
        if self.bridge is None:
            self.bridge = BridgeHoleDesign()
        
        ## Set up reward range from mass.
        # mass = l*h - hole_area; and reward = max_mass - current_mass
        assert(length == bridge.l)
        assert(height == bridge.h)
        max_mass = length*height
        self.reward_range = (0, max_mass)
        
        ## Set up action space
        nb_rld = len(bridge.rld)
        max_radius = np.sqrt(self.length**2 + self.height**2)
        self.action_space = gym.spaces.Box(low = 0, high = max_radius, shape = (nb_rld,))
        
        ## Set up observation space
        self.observation_space = None # TODO
        
        raise NotImplementedError
    
    def step(self, action):
        # action = vector
        raise NotImplementedError
    
    def reset(self):
        raise NotImplementedError

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed=123456789):
        raise NotImplementedError
    

    
        
