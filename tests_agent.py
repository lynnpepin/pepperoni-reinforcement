"""tests_agent.py

Unit tests for agent.py
"""

import unittest
import numpy as np
from random import random
from reinforcement_utils import State, state_from_update, _preprocess, _normalize_01, _normalize_angle
from agent import BasicAgent
from example_update_data import example_update_1
from random import randint, seed

class BasicAgent_Superficial_Tests(unittest.TestCase):
    """Check that the BasicAgent has methods that can be called, that setters
    and getters perform their functions properly, and do 'type-checking'.
    Does not check behavior (e.g. that rld does not violate boundaries)."""
    
    def setUp(self):
        # initialize an empty BasicAgent to begin with, each time
        random.seed(413)
        pass

    def tearDown(self):
        pass
    
    def test_(self):
        raise NotImplementedError

    def test_attr_initial_state(self):
        # is None if not provided, is state (and instance_of bridgestate) if provided
        raise NotImplementedError

    def test_attr_initial_state_if_provided(self):
        raise NotImplementedError

    def test_attr_initial_rld_is_none_if_not_provided(self):
            # Initial rld is None if not provided,
        raise NotImplementedError

    def test_attr_initial_rld_versus_stateself):
            #   = state.rld if state rld provided but rld not provided,
            #   = rld if state.rld provided and rld provided,
            #   = rld if rld provided but state not provided
        raise NotImplementedError

    def test_attr_initial_environment(self):
    # Initial environment is None if not provided
    #   = instance of BHD is provided
        raise NotImplementedError

    def test_attr_initial_preprocess(self):
        # Preprocess is preprocess_bridge_state is none provided
        #   = Provided function otherwises
        raise NotImplementedError
    
    def test_iterations(self):
        # raw_update is none to begin with
        # test the behavior of a whole iteration
        raise NotImplementedError
        
    def test_step_1(self):
        raise NotImplementedError
        
    def test_step_2(self):
        raise NotImplementedError
        
    def test_step_3(self):
        raise NotImplementedError
        
    def test_step_123(self):
        # should be equivalent to iterate
        raise NotImplementedError
        
    def test_step_1s2s3s(self):
        raise NotImplementedError
        for i in range(100):
            steps_1 = randint(1,10)
            steps_2 = randint(1,10)
            steps_3 = randint(1,10)
            # TODO - perform steps
    
            
TestCases = [BasicAgent_Superficial_Tests]


def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)

