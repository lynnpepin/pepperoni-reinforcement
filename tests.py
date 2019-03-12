import unittest
import numpy as np
from reinforcement_utils import State#, _read_update_data()

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
    '''
    # Test in progress...
    def setUp(self):
        self.example_update = {
            'r'         : ,
            'rb'        : ,
            'sigma'     : ,
            'mass'      : ,
            'gmass_r'   : ,
            'gmass_rld' : ,
            'geometry_info' : {
                'angles_ld'         : ,
                'angles_cb'         : ,
                'total_length_ld'   : ,
                'total_length_cb'   : ,
                'edges_ld'          : ,
                'edges_cb'          : ,
                'positions_ld'      : ,
                'positions_cb'      : ,
                'positions_all'     :
            }
        }
               
        pass
        
    def tearDown(self):
        pass

    def test_example(self):
        pass
    '''


TestCases = [StateEnvironmentalTests]

def run_tests(TestCaseList):
    for testcase in TestCaseList:
        Suite = unittest.TestLoader().loadTestsFromTestCase(testcase)
        unittest.TextTestRunner(verbosity=2).run(Suite)

if (__name__ == "__main__"):
    run_tests(TestCases)
