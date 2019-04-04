"""tests_reinforcement_utils.py

Unit tests for reinforcement_utils.py.
"""

import unittest
import numpy as np
from random import random
from gym_wrappers import BHDEnv, observe_bridge_update, observation_space_box, _normalize_01, _normalize_angle
from pepperoni import BridgeHoleDesign
from gym.spaces import Box, Dict, Discrete, MultiBinary, MultiDiscrete, Tuple
from collections import OrderedDict

class PreprocessUtilsTest(unittest.TestCase):
    # Preproc tests with _normalize_01, _normalize_angle
    def test_normalize_01_simple(self):
        """Test some simple cases of 0-1 normalization."""
        self.assertTrue(isinstance(_normalize_01(7,a=3,b=10), (np.ndarray, int, float)))
        self.assertAlmostEqual(_normalize_01(3), 3)
        self.assertAlmostEqual(_normalize_01(2.0), 2.0)
        self.assertAlmostEqual(_normalize_01(1.0), 1.0)
        self.assertAlmostEqual(_normalize_01(-3), -3)
        self.assertAlmostEqual(_normalize_01(-2.0), -2.0)
        self.assertAlmostEqual(_normalize_01(-1), -1)
        self.assertAlmostEqual(_normalize_01(0.0), 0.0)
        self.assertAlmostEqual(_normalize_01(7, b=10), .7)
        self.assertAlmostEqual(_normalize_01(-2, b=10, a=-10), .4)
        self.assertAlmostEqual(_normalize_01(.9, a=.5), .8)

    def test_normalize_01_vector(self):
        """Test that normalizing a vector produces the correct shape."""
        x = np.array([[-3, 10, 7],[0,-5,15]])
        np.testing.assert_array_almost_equal(_normalize_01(x, b=15, a=-5),
                                     np.array([[.1, .75, .6],[.25, 0, 1]]))

    def test_normalize_angle_right_angles(self, iterations=1):
        # todo - Should have used np testing before...
        """Test that rig normalizing right angles produces the correct
        (cos(x),sin(x)) array."""
        for k in range(-iterations,iterations):
            np.testing.assert_array_almost_equal(_normalize_angle(((2*k+0)*np.pi)), [1,0])
            np.testing.assert_array_almost_equal(_normalize_angle(((2*k+.5)*np.pi)), [0,1])
            np.testing.assert_array_almost_equal(_normalize_angle((2*k+1)*np.pi), [-1,0])
            np.testing.assert_array_almost_equal(_normalize_angle((2*k+1.5)*np.pi), [0,-1])
            np.testing.assert_array_almost_equal(_normalize_angle((4*k+0)*90, rad=False), [1,0])
            np.testing.assert_array_almost_equal(_normalize_angle((4*k+1)*90, rad=False), [0,1])
            np.testing.assert_array_almost_equal(_normalize_angle((4*k+2)*90, rad=False), [-1,0])
            np.testing.assert_array_almost_equal(_normalize_angle((4*k+3)*90, rad=False), [0,-1])
    
    def test_normalize_random_angles_nparray_and_unit(self, iterations=100):
        """Loop (default 100 times): Check that a random angle produces an
        np array of shape (2,) with unit length (1)"""
        for _ in range(iterations):
            pair = _normalize_angle(random()*360, rad=(random() >= .5))
            # We should always get a numpy array from this!
            self.assertTrue(isinstance(pair, np.ndarray))
            self.assertEqual(pair.shape, (2,))
            # And check that they are of length 1.
            self.assertAlmostEqual(np.linalg.norm(pair), 1)
            # Test a lot of random values, rad=True/False randomly make sure norm == 1
    
    def test_normalize_angle_right_angles_vector(self):
        """Test that normalize_angle on an nd array returns the appropriate shape."""
        x = _normalize_angle(np.array([-2*np.pi, -1.5*np.pi, -np.pi, -.5*np.pi,
                                       0, .5*np.pi, np.pi, 1.5*np.pi, 2*np.pi]))
        y = np.array([[1,0], [0,1], [-1,0], [0,-1], [1,0],
                      [0,1], [-1,0], [0,-1], [1,0]])
        np.testing.assert_array_almost_equal(x, y)
        
        x = _normalize_angle(np.array([[0, 90, 180], [270, -90, -180]]),
                             rad = False)
        y = np.array([[[1,0], [0,1], [-1,0]],
                      [[0,-1], [0,-1], [-1,0]]])
        np.testing.assert_array_almost_equal(x,y)

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
        ld_length = 10
        
        self.assertEqual(ld_length, len(bridge.rld))
        self.assertTrue(isinstance(obs, np.ndarray))
        self.assertTrue(obs.shape == (4 + 3*ld_length,))
        # points is (ld_length * 2), gmass_rld is ld_length
        # + 4 for stress, mass, and the provided stress_ratio, mass_ratio
        
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
            self.assertTrue(ob_space.shape == obs.shape)
            self.assertTrue(ob_space.contains(obs))
        
        # Finally, random numpy vectors should be inside the space
        
        for ii in range(10):
            print(ii)
            random_vec = 2*(np.random.rand(3*ld_length + 4) - .5)
            self.assertTrue(ob_space.contains(random_vec))


    def test_BHDEnv_init(self):
        bridge_env = BHDEnv(bridge=None, length = 20, height = 10, allowable_stress = 200.0)
    
    def test_BHDEnv_seed_and_reset(self):
        bridge_env = BHDEnv(bridge=None, length = 20, height = 10, allowable_stress = 200.0)
        bridge_env.seed(123456789)
        ob = bridge_env.reset()
    
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

        ob = bridge_env.reset()
        # Resetting bridge_env after step means rld should be the same)
        self.assertTrue(bridge_env.bridge.rld == rld)
        bridge_env.seed(612)
        bridge_env.seed(413)
        bridge_env.seed(1025)
        bridge_env.close()


TestCases = [UtilsTests,
             PreprocessUtilsTest]

def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)
