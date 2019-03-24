"""tests_agent.py

Unit tests for agent.py
"""

import unittest
import numpy as np
from random import random
from reinforcement_utils import State, state_from_update, _preprocess, _normalize_01, _normalize_angle
from agent import BasicAgent
from example_update_data import example_update_1

class BasicAgent_Superficial_Tests(unittest.TestCase):
    """Check that the BasicAgent has methods that can be called, that setters
    and getters perform their functions properly, and do 'type-checking'.
    Does not check behavior (e.g. that rld does not violate boundaries)."""
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_(self):
        raise NotImplementedError

    # Tests TODO
    # Test initial state is None if not provided, is state if provided
    # Initial rld is None if not provided,
    #   = state.rld if state rld provided but rld not provided,
    #   = rld if state.rld provided and rld provided,
    #   = rld if rld provided but state not provided
    # Initial environment is None if not provided
    #   = instance of BHD is provided
    # Preprocess is preprocess_bridge_state is none provided
    #   = Provided function otherwises
    
    # Test iterations:
    #   _raw_update is None to begin with, is not None afterwards
    # Test step_1, step_2, and step_3 do what they are meant to do, in order.
    #   Test what they do when run in other orders.
    # Test iterate on it's own
    
            
TestCases = [BasicAgent_Superficial_Tests]


def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)

