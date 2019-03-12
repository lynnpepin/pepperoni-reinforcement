"""tests_reinforcement_utils.py

Unit tests for reinforcement_utils.py.
"""

import unittest
import numpy as np
from reinforcement_utils import State, state_from_update

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
                        grad_mass   = np.array([3,2,-1,2]),
                        grad_mass_ld = np.array([3,2,-1]),
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
        self.assertTrue(np.all(state.get_grad_mass()    == np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_grad_mass_ld() == np.array([3,2,-1])))
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
        self.assertTrue(np.all(state.get_grad_mass()    == np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_grad_mass_ld() == np.array([3,2,-1])))
        self.assertTrue(np.all(state.get_angles_ld()    == np.array([1.1, 1.9, -1.1, 0.2])))
        self.assertTrue(np.all(state.get_angles_accb()  == np.array([1.0, 2.2, -1.1, 0.0])))
        self.assertTrue(np.all(state.get_edge_lengths_ld() == np.array([2,4,2,3])))
        self.assertTrue(np.all(state.get_edge_lengths_accb() == np.array([2,1,2,3])))
        self.assertTrue(np.all(state.get_points_ld()    == np.array([[1,1],[0,0],[-1,1]])))
        self.assertTrue(np.all(state.get_points_accb()  == np.array([[1,1],[0,0],[-.5,.5]])))
        self.assertTrue(np.all(state.get_points_ci()    == np.array([[1,1],[0,0],[-2,2]])))

    def test_init(self):
        """Ensure init works, and that set_state still works afterwards."""
        state = State(rld          = np.array([1,2,3,4]),
                      raccb        = np.array([5,6]),
                      ri           = np.array([7,8,9,10,1,2,3]),
                      stress        = 100,
                      mass          = 200,
                      grad_mass     = np.array([3,2,-1,2]),
                      grad_mass_ld  = np.array([3,2,-1]),
                      angles_ld     = np.array([1.0, 2.0, -1.0, 0.0]),
                      angles_accb   = np.array([1.0, 2.2, -1.0, 0.0]),
                      edge_lengths_ld  = np.array([2,4,2,3]),
                      edge_lengths_accb = np.array([2,1,2,3]),
                      points_ld     = np.array([[1,1],[0,0],[-1,1]]),
                      points_accb   = np.array([[1,1],[0,0],[-.5,.5]]),
                      points_ci     = np.array([[1,1],[0,0],[-2,2]]))
        state._set_state(stress = 60,
                         angles_ld = np.array([1.1, 1.9, -1.1, 0.2]),
                         mass   = 200)
        self.assertTrue(np.all(state.get_rld()      == np.array([1,2,3,4])))
        self.assertTrue(np.all(state.get_raccb()    == np.array([5,6])))
        self.assertTrue(np.all(state.get_ri()       == np.array([7,8,9,10,1,2,3])))
        self.assertEqual(state.get_stress(), 60)
        self.assertEqual(state.get_mass(), 200)
        self.assertTrue(np.all(state.get_grad_mass()    == np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_grad_mass_ld() == np.array([3,2,-1])))
        self.assertTrue(np.all(state.get_angles_ld()    == np.array([1.1, 1.9, -1.1, 0.2])))
        self.assertTrue(np.all(state.get_angles_accb()  == np.array([1.0, 2.2, -1.0, 0.0])))
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
            self.assertRaises(TypeError, f, ("Some String!",))
            self.assertRaises(TypeError, f, (None,))
            self.assertRaises(TypeError, f, (State(),))
        for f in [state._set_rld, state._set_raccb, state._set_ri,
                  state._set_grad_mass, state._set_grad_mass_ld,
                  state._set_angles_ld, state._set_angles_accb,
                  state._set_edge_lengths_ld, state._set_edge_lengths_accb]:
            self.assertRaises(TypeError, f, (1234,))
            self.assertRaises(TypeError, f, (self.a_2by4_matrix,))
            
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
        state = State(rld          = np.array([1,2,3,4]),
                      raccb        = np.array([5,6]),
                      ri           = np.array([7,8,9,10,1,2,3]),
                      stress        = 100,
                      mass          = 200,
                      grad_mass     = np.array([3,2,-1,2]),
                      grad_mass_ld  = np.array([3,2,-1,2]),
                      angles_ld     = np.array([1.0, 2.0, -1.0]),
                      angles_accb   = np.array([1.0, 2.0, -1.0]),
                      edge_lengths_ld  = np.array([2,4,2,3]),
                      edge_lengths_accb = np.array([2,1,2,3]),
                      points_ld     = np.array([[1,1],[0,0],[-1,1]]),
                      points_accb   = np.array([[1,1],[0,0],[-.5,.5]]),
                      points_ci     = np.array([[1,1],[0,0],[-2,2]]))
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
        self.assertRaises(ValueError, state._set_grad_mass_ld,  np.array([3,2,0]))
        self.assertRaises(ValueError, state._set_angles_ld,     np.array([1.0, 2.0, -1.0, 0.0]))
        self.assertRaises(ValueError, state._set_angles_accb,   np.array([1.0, 2.0, -1.0, 0.0]))
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
        # from pepperoni import BridgeHoleDesign as BHD; my_bridge = BHD();
        # rld = my_bridge.rld; data = my_bridge.update(rld)

        self.data1 = {
            'raccb'     : [0.78637574908387, 0.30650510646465656, 0.8333341891282224, 0.32905813582954196, 0.7552393486837017,
                           0.5910110035709384, 0.28963873290829917, 0.8464361369979728, 0.5063532680048215, 0.22494013807272537,
                           0.6966162536315875, 0.5008139027406175, 0.25936448795324213, 0.5414412724647085, 0.6758929339601379,
                           0.21624981762187698, 0.5117491630717859, 0.8351695447037216, 0.557150063306941, 0.23585903811584966,
                           0.5904202879724214],
            'ri'        : [0.5611760201979079, 0.26984139859763084, 0.7958691069422932, 0.4499041426801013, 0.6800964877159675,
                           0.33718591264397596, 0.601144080933035, 0.5469922363288172, 0.5429594321064543, 0.540143370084626,
                           0.4564296352362881, 0.9547359357092762, 0.5019169065205032, 0.7428058545198343, 0.3862305583120833,
                           0.3944271660557279, 0.7634401402443293, 0.3585379343291296, 0.4922735581461207, 0.23842012116032646,
                           0.4971215567310731, 0.6756226290662349, 0.2513469735147221, 0.5049374612367055, 0.42370375717395564,
                           0.6286877155817446, 0.6024680537515983, 0.23407330272450166, 0.5337003827357276, 0.385304716169418,
                           0.7745132176837706, 0.5357323034599337, 0.3756412192801929, 0.47638891191725813, 0.6070642560175765,
                           0.25688650103622107, 0.5256494770329644, 0.5390641632635531, 0.4011897493573875, 0.7555415043527308,
                           0.4982164284052373, 0.5015155329025384, 0.36798216725798627, 0.38661698281685414, 0.7863757490838704,
                           0.2722809333482741, 0.5360538392941846, 0.4688273419264768, 0.6470571791828162, 0.5263859016802367,
                           0.5992234936539511, 0.31225945303210867, 0.6750823484394433, 0.4119222079197042, 0.44064278584817823,
                           0.38855490902718975, 0.6754875184973432, 0.299895598909731, 0.7036173661839118, 0.25049384494234894,
                           0.5618498360036128, 0.6978712930092489, 0.3143900541878134, 0.8009790061141119, 0.5653441279700634,
                           0.8486397355169775, 0.7286800201345498], 
            'sigma'     : 189.25177318820099,
            'mass'      : 98.51585460574245,
            'gmass_r'   : [-1.00838458428583, -2.1550878225988495, -2.129947900713745, -2.2490368213003347, -2.3177079234170996,
                           -2.1031617445608175, -2.2252939777457232, -1.1729567124439737, -2.30676547230191, -4.1171677101397135,
                           -3.769729616900538, -4.428701803093125, -4.104041195731248, -4.305066311687599, -4.659565652370226,
                           -2.4823716749368727, -1.895725289671765, -4.242206894485023, -3.9847979625820615, -3.9098187412606817,
                           -3.867170819569327, -4.213192432280541, -5.708078700561973, -3.3349839852291217, -1.5591242706624096,
                           -3.707919716467353, -4.084526229862462, -3.498659822045894, -3.593828972033249, -4.297345884385294,
                           -5.8461450467660265, -3.9240775388485902, -1.9042784751204533, -3.6118456115772646, -3.3180376188577876,
                           -3.6618142880129785, -3.791731191454098, -3.510246818261183, -4.408927723112929, -1.8430996613243495,
                           -3.7983394395316408, -3.9120977717423604, -3.7329906717744863, -3.240886310298063, -3.338795139988228,
                           -4.381111458570259, -1.814863870281517, -4.240062490183854, -3.8447893961216146, -3.390126440116621,
                           -3.5223291378288115, -3.759578675983291, -4.654311251117499, -1.975821228491216, -3.7799885736149434,
                           -3.8688718897156327, -3.697642231172302, -4.171362868788732, -5.129029154318879, -6.684682659175502,
                           -1.8614836704006148, -3.316923990551234, -3.5447349768793726, -4.335227254851508, -3.794475378424467,
                           -4.632348079113859, -4.0701141065674005, -1.498550220439805, -3.4892988488748458, -3.971732226339096,
                           -3.792759371657852, -3.742642421799968, -5.265690765667973, -1.9450602747479486, -4.16100185964782,
                           -3.70062245129494, -3.1836773875581454, -3.4700933770162594, -6.15046992023221, -2.313754548139684,
                           -4.115853361519603, -3.9644510550666765, -3.5613773539083793, -4.738008361730602, -2.0798801950645727,
                           -4.380136182571227, -4.317015228301413, -5.51371016908106, -3.559865931585143, -1.610971026684044,
                           -4.231353607744108, -5.251968193075523, -3.192469275845865, -2.269918001464226, -5.170606082825159,
                           -2.717644734177225, -1.4575185751164155, -1.9855404000660464],
            'gmass_rld' : [-1.4575185751164155, -1.9855404000660464, -2.717644734177225, -3.192469275845865, -3.559865931585143,
                           -4.0701141065674005, -3.9240775388485902, -3.3349839852291217, -2.4823716749368727, -1.1729567124439737],
            'geometry_info' : {
                'angles_ld'         : [1.5693879345793422, 2.352233147955184, 3.140523099589756, 3.143075197060534, 2.8154115243187623,
                                       2.8783167105726855, 2.940563056123495, 3.143472405006232,3.1447903630881964, 1.5716835478575115],
                'angles_accb'       : [3.141538935170064, 3.141541649615242, 3.1417891900198573, 3.1415636845362327, 3.141881295914901,
                                       3.1417254293084866, 1.5708184228677482, 3.1416795738979264, 3.1416325842260036, 3.141811743519036, 
                                       3.1416138684055994, 3.1418273172974454, 3.141678709267026, 3.1414945114229758, 3.141418333669131, 
                                       3.1406374004311983, 3.1417589520252243, 3.1413903876305147, 3.141581550387412, 3.141500233948007, 
                                       3.1415854963252614],
                'total_length_ld'   : 19.53680502290873,
                'total_length_accb' : 20.802441111518995,
                'edges_ld'          : [1.7388962707751554, 1.43302690924238, 1.674485632027098, 2.3986980679402317, 3.465807711272462, 
                                       3.9006577793194763, 2.3879468195061744, 1.3221447258506975, 1.2151411069750506],
                'edges_accb'        : [1.0928808555485265, 1.1398392955928789, 1.1623923249577643, 1.0842974845132436, 1.34625035225464,
                                       0.8806497364792376, 1.136074869906272, 1.3527894050027944, 0.7312934060775469, 0.9215563917043129,
                                       1.1974301563722052, 0.7601783906938597, 0.8008057604179506, 1.2173342064248465, 0.892142751582015,
                                       0.7279989806936629, 1.3469187077755076, 1.3923196080106626, 0.7930091014227906, 0.826279326088271],
                'positions_ld'      : np.array([[15.77198526,  0.        ], [15.76953621,  1.73889455], [14.75080663,  2.74674184],
                                                [13.55890934,  3.9229468 ], [11.85139143,  5.5982511 ], [ 8.73813782,  7.1212334 ], 
                                                [ 4.90892497,  7.86437538], [ 2.53727502,  7.84130856], [ 1.21514063,  7.83608004],
                                                [ 0.        ,  7.83500195]]),
                'positions_accb'    : np.array([[0., 6.70631005], [0., 5.61342919], [0., 4.4735899], [0., 3.31119757], [0., 2.22690009],
                [0., 0.88064974], [0., 0.], [1.13607487, 0.], [2.48886427, 0.], [3.22015768, 0.], [4.14171407, 0.], [5.33914423, 0.],
                [6.09932262, 0.], [6.90012838, 0.], [8.11746259, 0.], [9.00960534, 0.], [9.73760432, 0.], [11.08452303, 0.], [12.47684263, 0.],
                [13.26985174, 0.], [14.09613106, 0.]])
,
                'positions_ci'      : np.array([[ 1.02687528, 1.40320318], [ 1.02505462, 2.23421861], [ 1.12481044, 3.29498697],
                                                [ 1.28197326, 4.53055635], [ 0.9865705 , 5.62126192], [ 1.12094958, 6.62974081], [ 2.1336349 , 1.04898163],
                                                [ 1.83831711, 2.15792104], [ 2.44086274, 3.06413168], [ 2.18014851, 4.1140527 ], [ 1.99279022, 5.09285535],
                                                [ 2.39943354, 6.44388919], [ 3.18820392, 0.72615434], [ 3.11325494, 1.96861858], [ 3.37004569, 3.06774742],
                                                [ 3.04757526, 3.77868997], [ 3.19422389, 4.92723291], [ 3.6119847 , 5.96834981], [ 4.06927386, 1.18668083],
                                                [ 4.09306134, 1.91698721], [ 4.14860591, 2.65042866], [ 4.11671801, 3.82273924], [ 4.19287524, 4.74657511],
                                                [ 4.33095139, 5.49014833], [ 4.90761386, 0.81762733], [ 4.95882764, 1.86877192], [ 5.15932511, 3.08349217],
                                                [ 5.02233951, 3.90874146], [ 4.97465595, 4.67503299], [ 5.21835875, 5.56113644], [ 6.08615063, 1.03320394],
                                                [ 6.02278642, 2.3419164 ], [ 6.12355164, 3.24770229], [ 5.72677829, 4.0010592 ], [ 6.05804794, 5.03225248],
                                                [ 5.79015198, 5.8534754 ], [ 7.38351292, 0.95132647], [ 7.02452204, 1.95241264], [ 6.79981087, 2.86541992],
                                                [ 6.95814675, 4.01126326], [ 7.1416622 , 5.24931416], [ 6.5260147 , 6.03699566], [ 8.27348208, 1.03214977],
                                                [ 7.91230868, 1.69470063], [ 7.98737417, 2.86313446], [ 7.98179656, 3.91801774], [ 8.03302298, 4.7247277 ],
                                                [ 9.0355552 , 0.68458551], [ 8.94329127, 1.79664919], [ 9.29693709, 2.90522652], [ 8.85303389, 3.93959285],
                                                [ 8.87190374, 4.85088045], [10.08744218, 1.13409987], [ 9.91864364, 2.19910313], [10.2613593 , 2.9769596 ],
                                                [ 9.80263502, 3.66771343], [ 9.85229946, 4.73059618], [11.06241993, 1.13472141], [11.03263289, 2.14031118],
                                                [10.953728  , 3.07668592], [10.73306494, 3.86154157], [12.05911246, 1.18346107], [12.04913383, 2.19567323],
                                                [11.98077618, 3.3020017 ], [13.26311833, 0.80117487], [13.21201536, 2.21424294], [14.47131393, 1.26461988]])
            }
        }
        
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
        self.assertTrue(np.all(state.get_points_ld()    == data['geometry_info']['positions_ld']))
        self.assertTrue(np.all(state.get_points_accb()  == data['geometry_info']['positions_accb']))
        self.assertTrue(np.all(state.get_points_ci()    == data['geometry_info']['positions_ci']))
        return True

    def test_example_1(self):
        self.data_state_equality_test(state = state_from_update(self.data1),
                                      data  = self.data1)
    
    # todo: More tests, on more examples, whne they become available.
    # todo: Test state_from_update(data, instance of state)


TestCases = [StateEnvironmentalTests,
             StateFromUpdateTest]

def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)
