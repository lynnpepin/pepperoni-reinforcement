"""tests_reinforcement_utils.py

Unit tests for reinforcement_utils.py.
"""

import unittest
import numpy as np
from random import random
from reinforcement_utils import State, state_from_update, _preprocess, _normalize_01, _normalize_angle
from example_update_data import example_update_1

class StateEnvironmentalTests(unittest.TestCase):
    """Test all the environmental/external values for the state,
    i.e. wholly ignoring any agent-internal state values."""
    def setUp(self):
        self.an_array = np.array([1, 2, 3.0])
        self.a_larger_array = np.array([1, 2, 3.0, 4.0, 5])
        self.a_matrix = np.array([[1, -3, 2.0], [0, 1, 2]])
        self.a_2by4_matrix = np.array([[0, 1], [1, 1], [-1, 1], [1, 0]])
        self.another_matrix = np.array([[0, 1], [1, 1], [-1, 1], [1, 0], [.5, .2]])
        self.some_angles=np.array([1.0, 2.0, -1.0, 0.0])
        self.other_angles=np.array([0.62, 2.2, -1.3, 1.0])
        
        pass
        
    def tearDown(self):
        pass
        
    def test_call(self):
        """Test that we can call each setter, getter, and init."""
        state = State()
        state._set_ri(np.array([0,1,2,3]))
        state._set_rld(np.array([4,5,6]))
        state._set_raccb(np.array([7,8,9]))
        state._set_stress(100)
        state._set_stress(101.0)
        state._set_mass(4)
        state._set_mass(4.0)
        state._set_grad_mass(np.array([.1,.1,.1,-.1,.2,.2,.2,.3,.3,.4]))
        state._set_grad_mass_ld(np.array([-.1,.4,.3]))
        state._set_angles_ld(self.some_angles)
        state._set_angles_accb(self.other_angles)
        state._set_edge_lengths_ld(np.array([1,2,3,4,5.0,6,7]))
        state._set_edge_lengths_accb(np.array([1,2.0,6,7]))
        state._set_points_ld(self.a_2by4_matrix)
        state._set_points_accb(self.another_matrix)
        state._set_points_ci(self.another_matrix)
        
        self.assertTrue(np.all(state.get_rld()          == state._rld))
        self.assertTrue(np.all(state.get_raccb()        == state._raccb))
        self.assertTrue(np.all(state.get_ri()           == state._ri))
        self.assertTrue(np.all(state.get_stress()       == state._stress))
        self.assertTrue(np.all(state.get_mass()         == state._mass))
        self.assertTrue(np.all(state.get_grad_mass()    == state._grad_mass))
        self.assertTrue(np.all(state.get_grad_mass_ld() == state._grad_mass_ld))
        self.assertTrue(np.all(state.get_angles_ld()    == state._angles_ld))
        self.assertTrue(np.all(state.get_angles_accb()  == state._angles_accb))
        self.assertTrue(np.all(state.get_edge_lengths_ld()   == state._edge_lengths_ld))
        self.assertTrue(np.all(state.get_edge_lengths_accb() == state._edge_lengths_accb))
        self.assertTrue(np.all(state.get_points_ld()        == state._points_ld))
        self.assertTrue(np.all(state.get_points_accb()      == state._points_accb))
        self.assertTrue(np.all(state.get_points_ci()        == state._points_ci))


    def test_change_ri(self):
        """Test that the setter properly changes r, scalar values."""
        state = State()
        
        state._set_rld(np.array([1,2,3,4]))
        self.assertTrue(np.all(state.get_rld() == np.array([1,2,3,4])))
        state._set_rld(np.array([100,2,3,10]))
        self.assertTrue(np.all(state.get_rld() == np.array([100,2,3,10])))
        
        state._set_raccb(np.array([1,2,3,4]))
        self.assertTrue(np.all(state.get_raccb() == np.array([1,2,3,4])))
        state._set_raccb(np.array([100,2,3,10]))
        self.assertTrue(np.all(state.get_raccb() == np.array([100,2,3,10])))
        
        state._set_ri(np.array([1,2,3,4]))
        self.assertTrue(np.all(state.get_ri() == np.array([1,2,3,4])))
        state._set_ri(np.array([100,2,3,10]))
        self.assertTrue(np.all(state.get_ri() == np.array([100,2,3,10])))
        
        state._set_stress(123.4)
        self.assertEqual(state.get_stress(), 123.4)
        state._set_stress(213)
        self.assertEqual(state.get_stress(), 213)
        
        state._set_mass(222)
        self.assertEqual(state.get_mass(), 222)
        state._set_mass(333)
        self.assertEqual(state.get_mass(), 333)

    def test_change_others(self):
        """Test that the setter properly changes the remaining values."""
        state = State()
        
        state._set_grad_mass(np.array([1,1,-1.2,1.1]))
        self.assertTrue(np.all(state.get_grad_mass() == np.array([1,1,-1.2,1.1])))
        state._set_grad_mass(np.array([2, 3.1, 4,1.1]))
        self.assertTrue(np.all(state.get_grad_mass() == np.array([2, 3.1, 4,1.1])))
        
        state._set_grad_mass_ld(np.array([1,1,-1.2,1.1]))
        self.assertTrue(np.all(state.get_grad_mass_ld() == np.array([1,1,-1.2,1.1])))
        state._set_grad_mass_ld(np.array([2, 3.1, 4,1.1]))
        self.assertTrue(np.all(state.get_grad_mass_ld() == np.array([2, 3.1, 4,1.1])))
        
        state._set_angles_ld(np.array([3.2, 3.1, 3.3]))
        self.assertTrue(np.all(state.get_angles_ld() == np.array([3.2, 3.1, 3.3])))
        state._set_angles_ld(np.array([3.2, 3.0, 3.3]))
        self.assertTrue(np.all(state.get_angles_ld() == np.array([3.2, 3.0, 3.3])))
        
        state._set_angles_accb(np.array([3.0, 3.1, 3.3]))
        self.assertTrue(np.all(state.get_angles_accb() == np.array([3.0, 3.1, 3.3])))
        state._set_angles_accb(np.array([3.2, 3.0, 3.9]))
        self.assertTrue(np.all(state.get_angles_accb() == np.array([3.2, 3.0, 3.9])))
        
        state._set_edge_lengths_ld(np.array([1.0, 1.2, .3]))
        self.assertTrue(np.all(state.get_edge_lengths_ld() == np.array([1.0, 1.2, .3])))
        state._set_edge_lengths_ld(np.array([1, 1, 1]))
        self.assertTrue(np.all(state.get_edge_lengths_ld() == np.array([1, 1, 1])))
        
        state._set_edge_lengths_accb(np.array([4.0, 3.3, .3]))
        self.assertTrue(np.all(state.get_edge_lengths_accb() == np.array([4.0, 3.3, .3])))
        state._set_edge_lengths_accb(np.array([4, 4, 4]))
        self.assertTrue(np.all(state.get_edge_lengths_accb() == np.array([4, 4, 4])))
        
        state._set_points_ld(np.array([[0, 1], [1,0], [1, 1]]))
        self.assertTrue(np.all(state.get_points_ld() == np.array([[0, 1], [1,0], [1, 1]])))
        state._set_points_ld(np.array([[0, 1], [1.2,0], [1.1, 3]]))
        self.assertTrue(np.all(state.get_points_ld() == np.array([[0, 1], [1.2,0], [1.1, 3]])))
        
        state._set_points_accb(np.array([[0, 1], [1,0], [1, 1]]))
        self.assertTrue(np.all(state.get_points_accb() == np.array([[0, 1], [1,0], [1, 1]])))
        state._set_points_accb(np.array([[0, 1], [1.2,0], [1.1, 3]]))
        self.assertTrue(np.all(state.get_points_accb() == np.array([[0, 1], [1.2,0], [1.1, 3]])))
        
        state._set_points_ci(np.array([[0, 1], [1,0], [1, 1]]))
        self.assertTrue(np.all(state.get_points_ci() == np.array([[0, 1], [1,0], [1, 1]])))
        state._set_points_ci(np.array([[0, 1], [1.2,0], [1.1, 3]]))
        self.assertTrue(np.all(state.get_points_ci() == np.array([[0, 1], [1.2,0], [1.1, 3]])))


    def test_change_set_state(self):
        """Ensure that set_state properly changes state values."""
        state = State()
        # 1. Init
        state._set_state(rld        = np.array([1,2,3,4]),
                        raccb       = np.array([5,6]),
                        ri          = np.array([7,8,9,10,1,2,3]),
                        stress      = 100,
                        mass        = 200,
                        grad_mass   = np.array([3,2,-1,2,5,6,7,8,9,10,1,2,3]),
                        grad_mass_ld = np.array([3,2,-1,2]),
                        angles_ld   = np.array([1.0, 2.0, -1.0, 0.0]),
                        angles_accb = np.array([1.0, 2.2, -1.1, 0.0]),
                        edge_lengths_ld   = np.array([2,4,2,3]),
                        edge_lengths_accb = np.array([2,1,2,3]),
                        points_ld   = np.array([[1,1],[0,0],[-1,1]]),
                        points_accb = np.array([[1,1],[0,0],[-.5,.5]]),
                        points_ci   = np.array([[1,1],[0,0],[-2,2]]))
        self.assertTrue(np.all(state.get_rld()      == np.array([1,2,3,4])))
        self.assertTrue(np.all(state.get_raccb()    == np.array([5,6])))
        self.assertTrue(np.all(state.get_ri()       == np.array([7,8,9,10,1,2,3])))
        self.assertEqual(state.get_stress(), 100)
        self.assertEqual(state.get_mass(), 200)
        self.assertTrue(np.all(state.get_grad_mass()    == np.array([3,2,-1,2,5,6,7,8,9,10,1,2,3])))
        self.assertTrue(np.all(state.get_grad_mass_ld() == np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_angles_ld()    == np.array([1.0, 2.0, -1.0, 0.0])))
        self.assertTrue(np.all(state.get_angles_accb()  == np.array([1.0, 2.2, -1.1, 0.0])))
        self.assertTrue(np.all(state.get_edge_lengths_ld() == np.array([2,4,2,3])))
        self.assertTrue(np.all(state.get_edge_lengths_accb() == np.array([2,1,2,3])))
        self.assertTrue(np.all(state.get_points_ld()    == np.array([[1,1],[0,0],[-1,1]])))
        self.assertTrue(np.all(state.get_points_accb()  == np.array([[1,1],[0,0],[-.5,.5]])))
        self.assertTrue(np.all(state.get_points_ci()    == np.array([[1,1],[0,0],[-2,2]])))
        # 2. Then change some values.
        # (Redo this, running init)
        state._set_state(stress         = 60,
                         angles_ld      = np.array([1.1, 1.9, -1.1, 0.2]),
                         angles_accb    = np.array([1.0, 2.2, -1.1, 0.0]),
                         mass           = 200)
        self.assertTrue(np.all(state.get_rld()      == np.array([1,2,3,4])))
        self.assertTrue(np.all(state.get_raccb()    == np.array([5,6])))
        self.assertTrue(np.all(state.get_ri()       == np.array([7,8,9,10,1,2,3])))
        self.assertEqual(state.get_stress(), 60)
        self.assertEqual(state.get_mass(), 200)
        self.assertTrue(np.all(state.get_grad_mass()    == np.array([3,2,-1,2,5,6,7,8,9,10,1,2,3])))
        self.assertTrue(np.all(state.get_grad_mass_ld() == np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_angles_ld()    == np.array([1.1, 1.9, -1.1, 0.2])))
        self.assertTrue(np.all(state.get_angles_accb()  == np.array([1.0, 2.2, -1.1, 0.0])))
        self.assertTrue(np.all(state.get_edge_lengths_ld() == np.array([2,4,2,3])))
        self.assertTrue(np.all(state.get_edge_lengths_accb() == np.array([2,1,2,3])))
        self.assertTrue(np.all(state.get_points_ld()    == np.array([[1,1],[0,0],[-1,1]])))
        self.assertTrue(np.all(state.get_points_accb()  == np.array([[1,1],[0,0],[-.5,.5]])))
        self.assertTrue(np.all(state.get_points_ci()    == np.array([[1,1],[0,0],[-2,2]])))

    def test_init(self):
        """Ensure init works, and that set_state still works afterwards."""
        state = State(rld         = np.array([1,2,3,4]),
                      raccb       = np.array([5,6]),
                      ri          = np.array([7,8,9,10,1,2,3]),
                      stress      = 100,
                      mass        = 200,
                      grad_mass   = np.array([3,2,-1,2,5,6,7,8,9,10,1,2,3]),
                      grad_mass_ld = np.array([3,2,-1,2]),
                      angles_ld   = np.array([1.0, 2.0, -1.0, 0.0]),
                      angles_accb = np.array([1.0, 2.2, -1.1, 0.0]),
                      edge_lengths_ld   = np.array([2,4,2,3]),
                      edge_lengths_accb = np.array([2,1,2,3]),
                      points_ld   = np.array([[1,1],[0,0],[-1,1]]),
                      points_accb = np.array([[1,1],[0,0],[-.5,.5]]),
                      points_ci   = np.array([[1,1],[0,0],[-2,2]]))
        state._set_state(stress = 60,
                         angles_ld = np.array([1.1, 1.9, -1.1, 0.2]),
                         mass   = 200)
        self.assertTrue(np.all(state.get_rld()      == np.array([1,2,3,4])))
        self.assertTrue(np.all(state.get_raccb()    == np.array([5,6])))
        self.assertTrue(np.all(state.get_ri()       == np.array([7,8,9,10,1,2,3])))
        self.assertEqual(state.get_stress(), 60)
        self.assertEqual(state.get_mass(), 200)
        self.assertTrue(np.all(state.get_grad_mass()    == np.array([3,2,-1,2,5,6,7,8,9,10,1,2,3])))
        self.assertTrue(np.all(state.get_grad_mass_ld() == np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_angles_ld()    == np.array([1.1, 1.9, -1.1, 0.2])))
        self.assertTrue(np.all(state.get_angles_accb()  == np.array([1.0, 2.2, -1.1, 0.0])))
        self.assertTrue(np.all(state.get_edge_lengths_ld() == np.array([2,4,2,3])))
        self.assertTrue(np.all(state.get_edge_lengths_accb() == np.array([2,1,2,3])))
        self.assertTrue(np.all(state.get_points_ld()    == np.array([[1,1],[0,0],[-1,1]])))
        self.assertTrue(np.all(state.get_points_accb()  == np.array([[1,1],[0,0],[-.5,.5]])))
        self.assertTrue(np.all(state.get_points_ci()    == np.array([[1,1],[0,0],[-2,2]])))


    def test_type_errors(self):
        """Ensure that setters only allow numbers/arrays"""
        # Check: String,  None, and state (float, array, matrix),
        state = State()
        for f in [state._set_rld, state._set_raccb, state._set_ri,
                  state._set_grad_mass, state._set_grad_mass_ld,
                  state._set_stress, state._set_mass,
                  state._set_angles_ld, state._set_angles_accb,
                  state._set_edge_lengths_ld, state._set_edge_lengths_accb,
                  state._set_points_ld, state._set_points_accb, state._set_points_ci]:
            self.assertRaises(TypeError, f, "Some String!")
            self.assertRaises(TypeError, f, None)
            self.assertRaises(TypeError, f, State())
        for f in [state._set_rld, state._set_raccb, state._set_ri,
                  state._set_grad_mass, state._set_grad_mass_ld,
                  state._set_angles_ld, state._set_angles_accb,
                  state._set_edge_lengths_ld, state._set_edge_lengths_accb]:
            self.assertRaises(TypeError, f, 1234)
            self.assertRaises(TypeError, f, np.array([[2,1,3],[4,1,2.1]]))
            
        for f in [state._set_stress, state._set_mass]:
            self.assertRaises(TypeError, f, (self.an_array,))
            self.assertRaises(TypeError, f, (self.a_2by4_matrix,))
        
        self.assertRaises(TypeError, state._set_points_ld, (1234,))
        self.assertRaises(TypeError, state._set_points_ld, (self.an_array,))
        
        self.assertRaises(TypeError, state._set_points_accb, (1234,))
        self.assertRaises(TypeError, state._set_points_accb, (self.an_array,))
        
        self.assertRaises(TypeError, state._set_points_ci, (1234,))
        self.assertRaises(TypeError, state._set_points_ci, (self.an_array,))

    def test_previous_size_errors(self):
        """Ensure providing arrays of different sizes throws a ValueError"""
        state = State(rld         = np.array([1,2,3,4]),
                      raccb       = np.array([5,6]),
                      ri          = np.array([7,8,9,10,1,2,3]),
                      stress      = 100,
                      mass        = 200,
                      grad_mass   = np.array([3,2,-1,2,5,6,7,8,9,10,1,2,3]),
                      grad_mass_ld = np.array([3,2,-1,2]),
                      angles_ld   = np.array([1.0, 2.0, -1.0, 0.0]),
                      angles_accb = np.array([1.0, 2.2, -1.1, 0.0]),
                      edge_lengths_ld   = np.array([2,4,2,3]),
                      edge_lengths_accb = np.array([2,1,2,3]),
                      points_ld   = np.array([[1,1],[0,0],[-1,1]]),
                      points_accb = np.array([[1,1],[0,0],[-.5,.5]]),
                      points_ci   = np.array([[1,1],[0,0],[-2,2]]))
        self.assertRaises(ValueError, state._set_rld,   (np.array([1,2,3,3,4])))
        self.assertRaises(ValueError, state._set_rld,   (np.array([1,2,3,3,4])))
        self.assertRaises(ValueError, state._set_rld,   (np.array([[1,2]])))
        self.assertRaises(ValueError, state._set_rld,   (np.array([[1,2]])))
        self.assertRaises(ValueError, state._set_raccb, (np.array([1,2,3,3,4])))
        self.assertRaises(ValueError, state._set_raccb, (np.array([1,2,3,3,4])))
        self.assertRaises(ValueError, state._set_raccb, (np.array([[1,2,3]])))
        self.assertRaises(ValueError, state._set_raccb, (np.array([[1,2,3]])))
        self.assertRaises(ValueError, state._set_ri,    (np.array([1,2,3,3,4,2,2,2,2,2])))
        self.assertRaises(ValueError, state._set_ri,    (np.array([1,2,3,3,4,2,2,2,2,2])))
        self.assertRaises(ValueError, state._set_ri,    (np.array([[1,2]])))
        self.assertRaises(ValueError, state._set_ri,    (np.array([[1,2]])))
        self.assertRaises(ValueError, state._set_grad_mass,     np.array([3,2,0]))
        self.assertRaises(ValueError, state._set_grad_mass,     np.array([3,2,0]))
        self.assertRaises(ValueError, state._set_grad_mass_ld,  np.array([3,2,0]))
        self.assertRaises(ValueError, state._set_grad_mass_ld,  np.array([3,2,0]))
        self.assertRaises(ValueError, state._set_angles_ld,     np.array([1.0, 2.0, -1.0, 0.0, 1.1]))
        self.assertRaises(ValueError, state._set_angles_ld,     np.array([1.0, 2.0, -1.0, 0.0, 1.1]))
        self.assertRaises(ValueError, state._set_angles_accb,   np.array([1.0, 2.0, -1.0,]))
        self.assertRaises(ValueError, state._set_angles_accb,   np.array([1.0, 2.0, -1.0,]))
        self.assertRaises(ValueError, state._set_edge_lengths_ld,   np.array([2,3,2,3,1]))
        self.assertRaises(ValueError, state._set_edge_lengths_accb, np.array([2,3,2,3,1]))
        self.assertRaises(ValueError, state._set_points_ld,     np.array([[1,1],[0,0],[-1,1],[-1,2]]))
        self.assertRaises(ValueError, state._set_points_accb,   np.array([[1,1],[0,0],[-1,1],[-1,2]]))
        self.assertRaises(ValueError, state._set_points_ci,     np.array([[1,1],[0,0],[-1,1],[-1,2]]))
    
    '''
        #def test_matching_size_errors(self):
            """Ensure providing logically mismatching sizes throws a ValueError"""
            # These tests were deprecated when moving from r, r_B --> ri, raccb, rld
            """
            # |r_B| <= |r|
            state1 = State()
            state1._set_r(np.array([1,2,3]))
            self.assertRaises(ValueError, state1._set_r_B, (np.array([1,2,3,4])))
            state2 = State()
            state2._set_r_B(np.array([1,2,3,4]))
            self.assertRaises(ValueError, state2._set_r, (np.array([1,2,3])))
            
            # |grad| == |r|
            self.assertRaises(ValueError, state1._set_grad_mass, (np.array([1,-2])))
            self.assertRaises(ValueError, state1._set_grad_mass, (np.array([1,-2,3,-4])))
            
            # todo - |grad_ld| == |r_ld|
            # TODO - define r_ld first
            
            # |E| >= |V|, <= 3|V|
            state3 = State()
            state3._set_points(np.array([[1,1],[0,0],[-1,1]]))
            self.assertRaises(ValueError, state3._set_edge_lengths, (np.array([1,2])))
            self.assertRaises(ValueError, state3._set_edge_lengths, (np.array([1,2,3,4,5,6,7,8,9,10])))
            state4 = State()
            state4._set_edge_lengths(np.array([1,2,3,4,5,6]))
            self.assertRaises(ValueError, state4._set_points, (np.array([[0,1]])))
            self.assertRaises(ValueError, state4._set_points, (np.array([[0,1],[-1,1],[-2,.5],[-3,.25],[1,1],[2,.5],[3,.25]])))
            """
            # todo - Consider relation between sizes of E, V, angles
    '''
    
    def test_misc_value_errors(self):
        """Test other value errors. E.g. Stress, mass, y >= 0..., |point|!=2"""
        state = State()
        # stress, mass, y >= 0
        self.assertRaises(ValueError, state._set_mass, -3)
        self.assertRaises(ValueError, state._set_stress, -10)
        self.assertRaises(ValueError, state._set_points_ld,     np.array([[1,-1]]))
        self.assertRaises(ValueError, state._set_points_accb,   np.array([[1,-1]]))
        self.assertRaises(ValueError, state._set_points_ci,     np.array([[1,-1]]))
        # |point| \in points == 2
        self.assertRaises(ValueError, state._set_points_ld,     np.array([[-1,-1,-3]]))
        self.assertRaises(ValueError, state._set_points_accb,   np.array([[-1,-1,-3]]))
        self.assertRaises(ValueError, state._set_points_ci,     np.array([[-1,-1,-3]]))
        # todo: Consider checking that ld, accb points are on borders, and non-overlapping
        #       for future tests using real outputs.
        # todo: points are R n*2, other vectors are R n, mass and stress are numbers


class StateInternalTests(unittest.TestCase):
    """Test all the internal state values."""
    def setUp(self):
        pass
        
    def tearDown(self):
        pass

    def test_example(self):
        pass

class StateFromUpdateTest(unittest.TestCase):
    """Test the _state_from_update(update(rld)) helper function."""
    
    # Test in progress...
    def setUp(self):
        self.data1 = example_update_1
        
    def tearDown(self):
        pass

    def data_state_equality_test(self, state, data):
        """Test all the attrs in data match those in state"""
        self.assertEqual(list(state.get_raccb()),               data['raccb'])
        self.assertEqual(list(state.get_ri()),	                data['ri'])
        self.assertEqual(state.get_stress(),                    data['sigma'])
        self.assertEqual(state.get_mass(),                      data['mass'])
        self.assertEqual(list(state.get_grad_mass()),           data['gmass_r'])
        self.assertEqual(list(state.get_grad_mass_ld()),        data['gmass_rld'])
        self.assertEqual(list(state.get_edge_lengths_ld()),	    data['geometry_info']['edges_ld'])
        self.assertEqual(list(state.get_edge_lengths_accb()),	data['geometry_info']['edges_accb'])
        self.assertEqual(list(state.get_angles_ld()),           data['geometry_info']['angles_ld'])
        self.assertEqual(list(state.get_angles_accb()),         data['geometry_info']['angles_accb'])
        self.assertTrue(np.all(state.get_points_ld()         == data['geometry_info']['positions_ld']))
        self.assertTrue(np.all(state.get_points_accb()       == data['geometry_info']['positions_accb']))
        self.assertTrue(np.all(state.get_points_ci()         == data['geometry_info']['positions_ci']))
        return True

    def test_example_1(self):
        """Test equality of example 1"""
        self.data_state_equality_test(state = state_from_update(self.data1),
                                      data  = self.data1)
    
    def test_returns_type(self):
        """Make sure it returns numbers or ndarrays."""
        state = state_from_update(self.data1)
        for getval in [state.get_rld, state.get_raccb, state.get_ri,
                       state.get_stress, state.get_mass,
                       state.get_grad_mass, state.get_grad_mass_ld,
                       state.get_edge_lengths_ld, state.get_edge_lengths_accb,
                       state.get_angles_ld, state.get_angles_accb,
                       state.get_points_ld, state.get_points_accb, state.get_points_ci]:
            value = getval()
            if value is not None:
                self.assertTrue(isinstance(value, (np.ndarray, int, float)))

    # todo: More tests, on more examples, whne they become available.
    # todo: Test state_from_update(data, instance of state)

class PreprocessTest(unittest.TestCase):
    """Ensure preprocess functions work as expected."""
    # _preprocess(l=20.0, w=10.0, max_mass=400.0, allowable_stress=200.0, state=None):
    # x, y, edge_lengths --> x/(sqrt(l**2+h**2))
    # angles --> (sin(x), cos(x))
    # gmass, gmass_r unmodified
    # sigma --> sigma/max_stress, mass --> mass/(2*l*h)
    # all should be in 0 to 1
    def setUp(self):
        pass
    
    def tearDown(self):
        pass
    
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
    
    def test_preprocess_example_1_sanitycheck(self):
        """Make sure _preprocess runs and produces a vector of shape (*,)"""
        state = state_from_update(example_update_1)
        vec   = _preprocess(state, l = 20, w = 10, max_mass = 400, allowable_stress = 200)
        # vec.shape should be of shape (n,)
        self.assertEqual(len(vec.shape), 1)
        # todo; more
        # todo: Consider specific tests on specific values?
        # todo: Check that, without a given key value, the preproccessor
        #       does not add it to the stack. E.g. what if ri is None

# class StatePreprocessorTest(unittest.TestCase)

TestCases = [StateEnvironmentalTests,
             StateFromUpdateTest,
             PreprocessTest]

def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)
