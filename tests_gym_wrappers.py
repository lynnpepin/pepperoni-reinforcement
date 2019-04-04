"""tests_reinforcement_utils.py

Unit tests for reinforcement_utils.py.
"""

import unittest
import numpy as np
from random import random
from reinforcement_utils import BridgeState, state_from_update, preprocess_bridge_state, _normalize_01, _normalize_angle
from gym_wrappers import BHDEnv, observe_bridge_update, observation_space_box
from pepperoni import BridgeHoleDesign
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from collections import OrderedDict

class UtilsTests(unittest.TestCase):
    """ Test utilities used for the gym_wrapper
    """
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_observe_bridge_update(self):
        # Make sure preprocess_update is an array of the correct shape
        bridge = BridgeHoleDesign()
        data = bridge.update(bridge.rld)
        obs = observe_bridge_update(data, length = bridge.l, height = bridge.h, allowable_stress=200.0)
        
        self.assertTrue(isinstance(obs, np.ndarray))
        self.assertTrue(obs.shape == (4 + 3*ld_length,))
        # points is (ld_length * 2), gmass_rld is ld_length
        # + 4 for stress, mass, and the provided stress_ratio, mass_ratio
        
        '''
        # For reference, these are old tests for a Dict implementation
        self.assertTrue(isinstance(obs, OrderedDict))
        self.assertTrue(isinstance(obs['mass'], np.ndarray))
        self.assertTrue(isinstance(obs['stress'], np.ndarray))
        self.assertTrue(obs['mass'].shape == obs['stress'].shape == (1,))
        self.assertEqual(len(data['gmass_rld']), len(obs['gmass_rld']))
        self.assertEqual(len(data['geometry_info']['positions_ld']), len(obs['points_ld']))
        self.assertTrue(isinstance(obs['mass_ratio'], np.ndarray))
        self.assertTrue(isinstance(obs['stress_ratio'], np.ndarray))
        self.assertTrue(obs['mass_ratio'].shape == obs['stress_ratio'].shape == (1,))
        self.assertTrue(0 <= obs['mass'] <= 1)
        self.assertTrue(0 <= obs['stress'] <= 1)
        self.assertTrue(0 <= obs['mass_ratio'] <= 1)
        self.assertTrue(0 <= obs['stress_ratio'] <= 1)
        '''
        
    
    def test_observation_space_Box(self):
        # Make sure the observation space is a box of the correct space
        ld_length = 10
        ob_space = observation_space_box(ld_length = ld_length)
        
        self.assertTrue(ob_space.shape == (4 + 3*ld_length,))
        self.assertTrue(np.all(ob_space.low == -1.0))
        self.assertTrue(np.all(ob_space.high == 1.0))
        
        # The observation should be inside its space!
        bridge = BridgeHoleDesign()
        rld = bridge.rld
        data = bridge.update(rld)
        obs = observe_bridge_update(data, length = bridge.l, height = bridge.h, allowable_stress=200.0)
        self.assertTrue(ob_space.contains(obs))
        
        # Increase 1 to a higher number if desired
        for ii in range(3):
            print(ii)
            bridge.update(rld)
            e = 0.001*np.random.rand(len(rld))
            data = bridge.update(rld + e)
            obs = observe_bridge_update(data, length = bridge.l, height = bridge.h, allowable_stress=200.0)
            self.assertTrue(ob_space.contains(obs))
        
        # Finally, random numpy vectors should be inside the space
        
        for ii in range(10):
            print(ii)
            random_vec = 2*(np.random.rand(ld_length) - .5)
            self.assertTrue(ob_space.contains(random_vec))
        
        '''
        ld_length = 10
        ob_space = observation_space_dict(ld_length = ld_length)
        somefloat = np.array([.55])
        somevec = np.random.rand(ld_length) - .5
        somepoints = np.random.rand(ld_length,2)
        # For reference, the old tests for a Dict implementation
        # It should be a Dict space
        self.assertTrue(isinstance(ob_space, Dict))
        # Test each space
        self.assertTrue(ob_space.spaces['mass'].contains(somefloat))
        self.assertTrue(ob_space.spaces['stress'].contains(somefloat))
        self.assertTrue(ob_space.spaces['mass_ratio'].contains(somefloat))
        self.assertTrue(ob_space.spaces['stress_ratio'].contains(somefloat))
        self.assertTrue(ob_space.spaces['gmass_rld'].contains(somevec))
        self.assertTrue(ob_space.spaces['points_ld'].contains(somepoints))
        
        # The observation should be inside its space!
        '''

    def test_BHDEnv_init(self):
        bridge_env = BHDEnv(bridge=None, length = 20, height = 10, allowable_stress = 200.0)
    
    def test_BHDEnv_seed_and_reset(self):
        bridge_env = BHDEnv(bridge=None, length = 20, height = 10, allowable_stress = 200.0)
        bridge_env.seed(123456789)
        ob = bridge_env.reset()

    def test_BHDEnv_keys(self):
        bridge_env = BHDEnv(bridge=None, length = 20, height = 10, allowable_stress = 200.0)
    
    def test_BHDEnv_trial(self):
        bridge_env = BHDEnv(bridge=None, length = 20, height = 10, allowable_stress = 200.0)
        rld = bridge_env.bridge.rld 
        del_rld = 0.001 * np.random.rand(len(rld))
        ob, reward, done, info = bridge_env.step(del_rld)
        self.assertTrue(isinstance(ob, np.ndarray))
        self.assertTrue(bridge_env.observation_space.contains(ob))
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, (bool, np.bool, np.bool8, np.bool)))
        self.assertTrue(isinstance(info, dict))
        
        # do some steps
        for ii in range(10):
            print(ii)
            rld = bridge_env.bridge.rld 
            del_rld = 0.001 * 2*(np.random.rand(len(rld)) - .5)
            ob, reward, done, info = bridge_env.step(del_rld)
        
        # Make sure each key is in the dict
        keys = ['mass', 'stress', 'gmass_rld', 'points_ld', 'mass_ratio', 'stress_ratio']
        for key in keys:
            self.assertTrue(key in ob.keys())
        
        ob = bridge_env.reset()
        # Resetting bridge_env after step means rld should be the same)
        self.assertTrue(bridge_env.bridge.rld == rld)
        for key in keys:
            self.assertTrue(key in ob.keys())
        
        bridge_env.seed(612)
        bridge_env.seed(413)
        bridge_env.seed(1025)
        bridge_env.close()


TestCases = [UtilsTests]

def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)
