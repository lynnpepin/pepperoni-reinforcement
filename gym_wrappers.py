# pep-action-space
# pep-observation-space
# BHDEnv

from pepperoni import BridgeHoleDesign
import numpy as np
import gym


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
        
