"""tests_reinforcement_utils.py

Unit tests for reinforcement_utils.py.
"""

import unittest
import numpy as np
from random import random
from reinforcement_utils import BridgeState, state_from_update, preprocess_bridge_state, _normalize_01, _normalize_angle
from gym_wrappers import BHDEnv, pep-action-space, pep-observation-space

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
        obs = preprocess_update(data, length = bridge.l, width = bridge.w, allowable_stress=200.0)
        
        #self.assertEqual(len(data['']), len(obs['']))
        self.assertEqual(len(data['stress']),       len(obs['stress']))
        self.assertEqual(len(data['mass']),         len(obs['mass']))
        self.assertEqual(len(data['gmass_rld']),    len(obs['grad_mass_ld']))
        self.assertEqual(len(data['geometry_info']['positions_ld']), len(obs['points_ld']))
        self.assertTrue(isinstance(obs['mass_ratio'], (float. int))
        self.assertTrue(isinstance(obs['stress_ratio'], (float. int))
        self.assertTrue(0 <= obs['mass_ratio'] <= 1)
        self.assertTrue(0 <= obs['stress_ratio'] <= 1)
        
        
        # TODO - assert shapes
        raise NotImplementedError

    
    def test_barebones(self):
        """Test the BHD can be instantiated"""
        bridge = BHDEnv(bridge=None, length = 20, height = 10, allowable_stress = 200.0)
        bridge.seed(123456789)
        observation = env.reset()
        # TODO - Get rld, from observation, do step
        rld = observation.rld
        env.step(rld + 0.001 * np.random.rand(len(rld)))
    
    def test_example(self):
        """Description"""
        raise NotImplementedError

TestCases = [UtilsTest]

def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)
