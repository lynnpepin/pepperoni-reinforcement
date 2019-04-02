"""tests_reinforcement_utils.py

Unit tests for reinforcement_utils.py.
"""

import unittest
import numpy as np
from random import random
from reinforcement_utils import BridgeState, state_from_update, preprocess_bridge_state, _normalize_01, _normalize_angle
from gymwrappers import BHDEnv, pep-action-space, pep-observation-space

class NewTest(unittest.TestCase):
    """ Empty test
    """
    
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
    def test_barebones(self):
        """Test the BHD can be instantiated"""
        bridge = BHDEnv(width = 20.0, height = 10.0, allowable_stress = 200.0)
        bridge.seed(123456789)
        observation = env.reset()
        # TODO - Get rld, from observation, do step
        rld = observation.rld
        env.step(rld + 0.001 * np.random.rand(len(rld)))
    
    def test_example(self):
        """Description"""
        raise NotImplementedError

TestCases = [NewTest]

def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)
