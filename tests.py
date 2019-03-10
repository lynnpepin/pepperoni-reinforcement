import unittest
import numpy as np
from reinforcement import State, _read_update_data()

class StateTests(unittest.TestCase):
    def setUp(self):
        self.an_array = np.array([1, 2, 3.0])
        self.a_larger_array = np.array([1, 2, 3.0, 4.0, 5])
        self.a_matrix = np.array([[1, -3, 2.0], [0, 1, 2]])
        self.a_2by4_matrix = np.array([[0, 1], [1, 1], [-1, -1], [1, 0]])
        self.some_angles=np.array([1.0, 2.0, -1.0, 0.0])
        
        pass
        
    def tearDown(self):
        pass
        
    def test_call(self):
        """Test that we can call each setter, getter, and init."""
        state = State()
        state._set_r_B(self.an_array)
        state._set_r(self.a_larger_array)
        state._set_stress(100)
        state._set_stress(101.0)
        state._set_mass(4)
        state._set_mass(4.0)
        state._set_grad_mass(self.a_larger_array)
        state._set_angles(self.some_angles)
        state._set_edge_lengths(self.an_array)
        state._set_points(self.a_2by4_matrix)
        
        self.assertEqual(state.get_r_B(),           state._r_B)
        self.assertEqual(state.get_r(),             state._r)
        self.assertEqual(state.get_stress(),        state._stress)
        self.assertEqual(state.get_mass(),          state._mass)
        self.assertEqual(state.get_grad_mass(),     state._grad_mass)
        self.assertEqual(state.get_angles(),        state._angles)
        self.assertEqual(state.get_edge_lengths(),  state._edge_lengths)
        self.assertEqual(state.get_points(),        state._points)
        
    def test_change(self):
        """Test that the setter properly changes values."""
        state = State()
        state._set_r_B(np.array([1,2,3]))
        self.assertTrue(np.all(state.get_r_B() == np.array([1,2,3])))
        state._set_r_B(np.array([10,2,3]))
        self.assertTrue(np.all(state.get_r_B() == np.array([10,2,3])))
        
        state._set_r(np.array([1,2,3,4]))
        self.assertTrue(np.all(state.get_r() == np.array([1,2,3,4])))
        state._set_r(np.array([100,2,3,10]))
        self.assertTrue(state.get_r() == np.array([100,2,3,10]))
        
        state._set_stress(123.4)
        self.assertEqual(state.get_stress(), 123.4)
        state._set_stress(213)
        self.assertEqual(state.get_stress(), 213)
        
        state._set_mass(222)
        self.assertEqual(state.get_mass(), 222)
        state._set_mass(333)
        self.assertEqual(state.get_mass(), 333)
        
        state._set_grad_mass(np.array([1,1,-1.2]))
        self.assertTrue(np.all(state.get_grad_mass() == np.array([1,1,-1.2])))
        state._set_grad_mass(np.array([2, 3.1, 4]))
        self.assertTrue(np.all(state.get_grad_mass() == np.array([2, 3.1, 4])))
        
        state._set_angles(np.array([3.2, 3.1, 3.3]))
        self.assertTrue(np.all(state.get_angles() == np.array([3.2, 3.1, 3.3])))
        state._set_angles(np.array([3.2, 3.0, 3.3]))
        self.assertTrue(np.all(state.get_angles() == np.array([3.2, 3.0, 3.3])))
        
        state._set_edge_lengths(np.array([1.0, 1.2, .3]))
        self.assertTrue(np.all(state.get_edge_lengths() == np.array([1.0, 1.2, .3])))
        state._set_edge_lengths(np.array([1, 1, 1]))
        self.assertTrue(np.all(state.get_edge_lengths() == np.array([1, 1, 1])))
        
        state._set_points(np.array([[0, 1], [1,0], [1, 1]]))
        self.assertTrue(np.all(state.get_points() == np.array([[0, 1], [1,0], [1, 1]])))
        state._set_points(np.array([[0, 1], [1.2,0], [1.1, 3]]))
        self.assertTrue(np.all(state.get_points() == np.array([[0, 1], [1.2,0], [1.1, 3]])))

    def test_change_set_state(self):
        """Ensure that set_state properly changes state values."""
        state = State()
        # 1. Init
        self.set_state(r_B      = np.array([1,2,3]),
                       r        = np.array([1,2,3,4]),
                       stress   = 100,
                       mass     = 200,
                       grad_mass    = np.array([3,2,-1,2]),
                       angles   = np.array([1.0, 2.0, -1.0, 0.0]),
                       edge_lengths = np.array([2,4,2,3]),
                       points   = np.array([[1,1],[0,0],[-1,1]]))
        self.assertTrue(np.all(state.get_r_B(), np.array([1,2,3])))
        self.assertTrue(np.all(state.get_r(), np.array([1,2,3,4])))
        self.assertEqual(state.get_stress(), 100)
        self.assertEqual(state.get_mass(), 200)
        self.assertTrue(np.all(state.get_grad_mass(), np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_angles(), np.array([1.0, 2.0, -1.0, 0.0])))
        self.assertTrue(np.all(state.get_edge_lengths(), np.array([2,4,2,3])))
        self.assertTrue(np.all(state.get_points(), np.array([[1,1],[0,0],[-1,1]])))
        # 2. Then change some values.
        # (Redo this, running init)
        self.set_state(stress = 60,
                       angles = np.array([1.1, 1.9, -1.1, 0.2]),
                       mass   = 200)
        self.assertTrue(np.all(state.get_r_B(), np.array([1,2,3])))
        self.assertTrue(np.all(state.get_r(), np.array([1,2,3,4])))
        self.assertEqual(state.get_stress(), 60)
        self.assertEqual(state.get_mass(), 200)
        self.assertTrue(np.all(state.get_grad_mass(), np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_angles(), np.array([1.1, 1.9, -1.1, 0.2])))
        self.assertTrue(np.all(state.get_edge_lengths(), np.array([2,4,2,3])))
        self.assertTrue(np.all(state.get_points(), np.array([[1,1],[0,0],[-1,1]])))

    def test_init(self):
        """Ensure init works, and that set_state still works afterwards."""
        state = State(r_B       = np.array([1,2,3]),
                      r         = np.array([1,2,3,4]),
                      stress    = 100,
                      mass      = 200,
                      grad_mass     = np.array([3,2,-1,2]),
                      angles    = np.array([1.0, 2.0, -1.0, 0.0]),
                      edge_lengths  = np.array([2,4,2,3]),
                      points    = np.array([[1,1],[0,0],[-1,1]]))
        self.set_state(stress = 60,
                       angles = np.array([1.1, 1.9, -1.1, 0.2]),
                       mass   = 200)
        self.assertTrue(np.all(state.get_r_B(), np.array([1,2,3])))
        self.assertTrue(np.all(state.get_r(), np.array([1,2,3,4])))
        self.assertEqual(state.get_stress(), 60)
        self.assertEqual(state.get_mass(), 200)
        self.assertTrue(np.all(state.get_grad_mass(), np.array([3,2,-1,2])))
        self.assertTrue(np.all(state.get_angles(), np.array([1.1, 1.9, -1.1, 0.2])))
        self.assertTrue(np.all(state.get_edge_lengths(), np.array([2,4,2,3])))
        self.assertTrue(np.all(state.get_points(), np.array([[1,1],[0,0],[-1,1]])))
    
    def test_type_errors(self):
        """Ensure that setters only allow numbers/arrays"""
        # Check: String, (float, array, matrix), None, and state
        state = State()
        self.assertRaises(TypeError, state.set_r_B())
        self.assertRaises(TypeError, state.set_r_B())
        self.assertRaises(TypeError, state.set_r())
        self.assertRaises(TypeError, state.set_r())
        self.assertRaises(TypeError, state.set_stress())
        self.assertRaises(TypeError, state.set_stress())
        self.assertRaises(TypeError, state.set_mass())
        self.assertRaises(TypeError, state.set_mass())
        self.assertRaises(TypeError, state.set_grad_mass())
        self.assertRaises(TypeError, state.set_grad_mass())
        self.assertRaises(TypeError, state.set_angles())
        self.assertRaises(TypeError, state.set_angles())
        self.assertRaises(TypeError, state.set_edge_lengths())
        self.assertRaises(TypeError, state.set_edge_lengths())
        self.assertRaises(TypeError, state.set_points())
        self.assertRaises(TypeError, state.set_points())

    def test_previous_size_errors(self):
        """Ensure providing arrays of different sizes throws a ValueError"""
        pass
    
    def test_matching_size_errors(self):
        """Ensure providing logically mismatching sizes throws a ValueError"""
        pass

    def test_misc_value_errors(self):
        """Test other value errors. E.g. Stress, mass, y >= 0..."""
        pass
        
    # then test errors
    
    # value errors:
    # |r| >= |r_B|
    # |r| == |r|_prev, also fo rr_B, grad_mass, angles, edge_lengths, points
    # |r| == |grad_mass|
    # |edge_lengths| <= (|points|**2 - |points|)/2
    # mass, stress >= 0
    # each point y >= 0 (?)
    
    # then, TODO - history? preproc? meta-values (l, w, allowable_stress, etc.?)
    # probably best to savef or later huh

if (__name__ == "__main__"):
    StateSuite = unittest.TestLoader().loadTestsFromTestCase(StateTests)
    unittest.TextTestRunner(verbosity=2).run(StateSuite)
