# pep-action-space
# pep-observation-space
# BHDEnv

from pepperoni import BridgeHoleDesign
from reinforcement_utils import _normalize_01, _normalize_angle
import numpy as np
from gym.spaces import Box, Dict
import gym
from collections import OrderedDict

def observe_bridge_update(data, length = 20.0, height = 10.0, allowable_stress=200.0):
    """Preprocess data and provide as an observation (dict of np array, shape (n,)).
    
    Arguments:
        data:  dict provided from BridgeHoleDesign.update(rld)
        length, width:  float >= 0
        allowable_stress:  maximum stress the bridge can take on
    
    Returns:
        dict of 'mass' --> np array shape (1,),
                'stress' --> np array shape (1,),
                'gmass_rld' --> np array shape (n,2), same shape as rld
                'points_ld' --> np array shape (n,)
                'mass_ratio' --> np array shape (1,),
                'stress_ratio' --> np array shape (1,),
        All values are preprocessed, normalized from 0 to 1 or -1, 1
    """
    max_radius = np.sqrt(length**2 + height**2)
    max_mass   = length*height
    gmass_rld  = np.array(data['gmass_rld'])
    out = OrderedDict()
    out['mass']        = np.array([_normalize_01(data['mass'], b = max_mass)])
    out['stress']      = np.array([_normalize_01(data['sigma'], b = allowable_stress)])
    out['gmass_rld']   = gmass_rld / np.linalg.norm(gmass_rld)
    out['points_ld']   = _normalize_01(np.array(data['geometry_info']['positions_ld']),
                                          b = max_radius)
    out['mass_ratio']  = np.array([(max_mass - data['mass'])/max_mass])
    out['stress_ratio']= np.array([(allowable_stress - data['sigma'])/allowable_stress])
    return out


def observation_space_dict(ld_length = 10):
    ''' A dictionary representing the observation space from `observe_bridge_update'.
    
    Arguments:
        ld_length:  integer >= 1, the number of leading dancers.
    Returns:  gym.spaces.Dict() of spaces:
        'mass', 'stress', 'mass_ratio', 'stress_ratio' --> Box(0, 1, [1])
        'points_ld' --> Box(0, 1, [ld_length,2])
        'gmass_rld' --> Box(-1, 1, [ld_length])
    '''
    out = { 'mass' : Box(low = 0.0, high = 1.0, shape = [1]),
            'stress' : Box(low = 0.0, high = 1.0, shape = [1]),
            'points_ld' : Box(low = 0.0, high = 1.0, shape = [ld_length, 2]),
            'gmass_rld' : Box(low = -1, high = 1, shape = [ld_length]),
            'mass_ratio' : Box(low = 0.0, high = 1.0, shape = [1]),
            'stress_ratio' : Box(low = 0.0, high = 1.0, shape = [1]),
          }
    return Dict(out)


class BHDEnv(gym.Env):
    """
    BridgeHoleDesign environment.
    See: https://github.com/openai/gym/blob/master/gym/core.py
        (Documentation shamelessly adapted from there.)
    
    Attributes:
        action_space: The Space object corresponding to valid actions
        observation_space: The Space object corresponding to valid observations
        reward_range: A tuple corresponding to the min and max possible rewards
        
    Methods:
        step:  Mapping action --> observation, reward, done, info 
        reset:  Returns observation of initial state of the space.
        render:  Renders the environment. Mode == 'human', 'rgb_array', or 'ansi' by convention.
        close:  Automatically run when garbage collection / exit.
        seed:  Set the seed for the environment's RNG/RNGs. Returns list of seed history.
    """
    
    def __init__(self, bridge = None,
                       length = 20.0,
                       height = 10.0,
                       allowable_stress = 200.0):
        
        self.__version__ = "0.1.0"
        
        # Set up bridge values
        self.length = length
        self.height = height
        self.allowable_stress = allowable_stress
        self.max_mass = self.length * self.height
        # self.reset sets self.bridge, used later
        self.reset(bridge = bridge)
        self.ld_length = len(self.bridge.rld)
        self.max_radius = np.sqrt(self.length**2 + self.height**2)
        
        # Set up values required for Env
        self.reward_range = (0, self.max_mass)
        self.action_space = gym.spaces.Box(low = 0, high = self.max_radius, shape = (self.ld_length,))
        self.observation_space = observation_space_dict(ld_length = self.ld_length)


    def step(self, action):
        """Accepts an action and returns a tuple (observation, reward, done, info).
        
        Argumentss:
            action (np.array): an action provided to the environment. I.e. rld
            
        Returns:
            observation (dict): agent's observation of the current environment.
                See observe_bridge_update
            reward (float) : reward returned: observation['mass_ratio']
            done (boolean): whether the episode has ended, in which case further
                step() calls will return undefined results.
                True if observation['stress'] >= allowable_stress.
            info (dict): auxiliary diagnostic information for debugging, logging.
        """
        # TODO - Tests
        raise NotImplementedError


    def reset(self, bridge=None):
        """Resets the state of the environment and returns an initial observation.
        Returns: observation (object): the initial observation of the space.
        """
        self.bridge = bridge
        if self.bridge is None:
            self.bridge = BridgeHoleDesign()
        
        data = self.bridge.update(self.bridge.rld)
        ob = observe_bridge_update(data, length = self.length, height = self.height,
                                   allowable_stress=self.allowable_stress)
        return ob


    def render(self, mode='human'):
        """Renders the environment.
        
        The set of supported modes varies per environment. (And some
        environments do not support rendering at all.)
        
        Conventional modes:
        - human: render to the current display or terminal.
        - rgb_array: Return an numpy.ndarray with shape (x, y, 3).
        - ansi: Return a string (str) or StringIO.StringIO containing a
          terminal-style text representation.
        
        Arguments:
            mode (str): the mode to render with
        """
        raise NotImplementedError


    def close(self):
        """Override _close in your subclass to perform any necessary cleanup.
        
        Environments will automatically close() themselves when
        garbage collected or when the program exits.
        """
        return


    def seed(self, seed=123456789):
        """Sets the seed for this env's random number generator(s).
        
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
            
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        """
        gym.logger.warn("Could not seed environment " + str(self))

    
        
