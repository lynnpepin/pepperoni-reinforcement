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
        
        self.assertTrue(isinstance(obs['mass'], (float, int)))
        self.assertTrue(isinstance(obs['stress'], (float, int)))
        self.assertEqual(len(data['gmass_rld']), len(obs['gmass_rld']))
        self.assertEqual(len(data['geometry_info']['positions_ld']), len(obs['points_ld']))
        self.assertTrue(isinstance(obs['mass_ratio'], (float, int)))
        self.assertTrue(isinstance(obs['stress_ratio'], (float, int)))
        self.assertTrue(0 <= obs['mass'] <= 1)
        self.assertTrue(0 <= obs['stress'] <= 1)
        self.assertTrue(0 <= obs['mass_ratio'] <= 1)
        self.assertTrue(0 <= obs['stress_ratio'] <= 1)
        
    
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
