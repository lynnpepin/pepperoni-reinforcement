"""tests_reinforcement_utils.py

Unit tests for reinforcement_utils.py.
"""

import unittest
import numpy as np
from random import random
from reinforcement_utils import BridgeState, state_from_update, preprocess_bridge_state, _normalize_01, _normalize_angle
from gym_wrappers import BHDEnv, observe_bridge_update, observation_space_dict
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
        # Make sure preprocess_update has the correct shapes
        bridge = BridgeHoleDesign()
        data = bridge.update(bridge.rld)
        obs = observe_bridge_update(data, length = bridge.l, height = bridge.h, allowable_stress=200.0)
        
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
        
    
    def test_observation_space_dict(self):
        ld_length = 10
        ob_space = observation_space_dict(ld_length = ld_length)
        somefloat = np.array([.55])
        somevec = np.random.rand(ld_length) - .5
        somepoints = np.random.rand(ld_length,2)
        
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
        bridge = BridgeHoleDesign()
        rld = bridge.rld
        data = bridge.update(rld)
        obs = observe_bridge_update(data, length = bridge.l, height = bridge.h, allowable_stress=200.0)
        self.assertTrue(ob_space.contains(obs))
        
        # Plus some random observations:
        for ii in range(3):
            print(ii)
            bridge.update(rld)
            e = 0.001*np.random.rand(len(rld))
            data = bridge.update(rld + e)
            obs = observe_bridge_update(data, length = bridge.l, height = bridge.h, allowable_stress=200.0)
            self.assertTrue(ob_space.contains(obs))
        
    
    def test_barebones(self):
        """Test the BHD Env can be instantiated and that all its functions work."""
        bridge = BHDEnv(bridge=None, length = 20, height = 10, allowable_stress = 200.0)
        bridge.seed(123456789)
        ob = env.reset()
        
        # Check ob is a dict with what we want
        keys = ['mass', 'stress', 'gmass_rld', 'points_ld', 'mass_ratio', 'stress_ratio']
        for key in keys:
            assertTrue(key in ob.keys())
        
        # TODO - Get rld, from observation, do step
        rld = bridge.rld 
        rld_new = 0.001 * np.random.rand(len(rld))
        ob, reward, done, info = env.step(rld_new)
        self.assertTrue(isinstance(ob, dict))
        self.assertTrue(isinstance(reward, float))
        self.assertTrue(isinstance(done, bool))
        self.assertTrue(isinstance(info, dict))
        
        for key in keys:
            assertTrue(key in ob.keys())
        
        ob = env.reset()
        keys = ['mass', 'stress', 'gmass_rld', 'points_ld', 'mass_ratio', 'stress_ratio']
        for key in keys:
            assertTrue(key in ob.keys())
            
        env.seed(612)
        env.seed(413)
        env.seed(1025)
        env.close()
        
        

TestCases = [UtilsTests]

def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)
