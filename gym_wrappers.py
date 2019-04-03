# pep-action-space
# pep-observation-space
# BHDEnv

from pepperoni import BridgeHoleDesign
from reinforcement_utils import _normalize_01, _normalize_angle
import numpy as np
from gym.spaces import Box, Dict
import gym

def observe_bridge_update(data, length = 20.0, height = 10.0, allowable_stress=200.0):
    """Preprocess data and provide as an observation (dict of np array, shape (n,)).
    
    Arguments:
        data:  dict provided from BridgeHoleDesign.update(rld)
        length, width:  float >= 0
        allowable_stress:  maximum stress the bridge can take on
    
    Returns:
        dict of 'mass' --> number,
                'stress' --> number,
                'mass_ratio' --> number,
                'stress_ratio' --> number,
                'gmass_rld' --> np array shape (n,2), same shape as rld
                'points_ld' --> np array shape (n,)
        All values are preprocessed, normalized from 0 to 1.
    """
    max_radius = np.sqrt(length**2 + height**2)
    max_mass   = length*height
    gmass_rld  = np.array(data['gmass_rld'])
    out = {'mass'         : _normalize_01(data['mass'],   b = max_mass),
           'stress'       : _normalize_01(data['sigma'], b = allowable_stress),
           'gmass_rld'    :  gmass_rld / np.linalg.norm(gmass_rld),
           'points_ld'    : _normalize_01(np.array(data['geometry_info']['positions_ld']),
                                          b = max_radius),
           'mass_ratio'   : (max_mass - data['mass'])/max_mass,
           'stress_ratio' : (allowable_stress - data['sigma'])/allowable_stress
          }
    
    return out


def observation_space_dict(ld_length = 10):
    out = { 'mass' : Box(low = 0.0, high = 1.0, shape = [1]),
            'stress' : Box(low = 0.0, high = 1.0, shape = [1]),
            'mass_ratio' : Box(low = 0.0, high = 1.0, shape = [1]),
            'stress_ratio' : Box(low = 0.0, high = 1.0, shape = [1]),
            'points_ld' : Box(low = 0.0, high = 1.0, shape = [ld_length]),
            'gmass_rld' : Box(low = -1, high = 1, shape = [ld_length])
          }
    return Dict(out)


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
        assert(length == self.bridge.l)
        assert(height == self.bridge.h)
        max_mass = length*height
        self.reward_range = (0, max_mass)
        
        ## Set up action space
        nb_rld = len(self.bridge.rld)
        max_radius = np.sqrt(length**2 + height**2)
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
    

    
        
