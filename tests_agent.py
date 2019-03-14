"""tests_agent.py

Unit tests for agent.py
"""

import unittest
import numpy as np
from random import random
from reinforcement_utils import State, state_from_update, _preprocess, _normalize_01, _normalize_angle
from agent import BasicAgent
from example_update_data import example_update_1

class BasicAgentTests(unittest.TestCase):
    """Test all the environmental/external values for the state,
    i.e. wholly ignoring any agent-internal state values."""
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_(self):
        raise NotImplementedError

TestCases = [BasicAgentTests]


def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)

