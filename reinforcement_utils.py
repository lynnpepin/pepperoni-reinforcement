import numpy as np


def _check_array_and_shape(x, new_x):
    """Check to ensure new_x is a numpy array with the same shape as x."""
    if not isinstance(new_x, np.ndarray):
        
        raise TypeError(str(type(new_x)) + ": Not a numpy array.")
    elif isinstance(x, np.ndarray):
        # if x is an array, make sure
        # x and new_x have matching shape.  
        if not(x.shape == new_x.shape):
            raise ValueError("Shape mismatch setting new value.")


def _check_number_and_geqzero(x):
    """Ensure x is a nubmer >= 0."""
    if not isinstance(x, (int, float)):
        raise TypeError("Not a number!")
    elif not (x > 0):
        raise ValueError("Not greater than zero!")


def _state_from_update(update_data, state = None):
    """Focal point from circlepacking/FEM code to RL side.
    
    Expected to be updated possibly often as the spec for update(rld_new) changes.
    Modifies and returns state if provided, or returns a new state, from the
    update data.
    
    # Arguments
        update_data, as provided by pepperoni.BridgeHoleDesign().update(r_B).
        Currently provided as a dict:
        {   'r'             : List of float,
            # MISSING - rld
            'sigma'         : Float, representing stress,
            'mass':         : Float,
            'gmass_r'       : List of float; gradient w.r.t. r of mass,
            'gmass_rld'     : List of float, ???,
            'geometry_info' : Dictionary, of format:
                                        # Lists of float:
                {   'angles_ld'         : Surround angles of leading dancers,
                    'angles_cb'         : Surround angles of boundary circles,
                                        # Float:
                    'total_length_ld'   : Total length of leading edges,
                    'total_length_cb'   : Total edge length of boundary,
                                        # List of float:
                    'edges_ld'          : Edge lengths of leading dancers,
                    'edges_cb'          : Edge lengths of boundary circles,
                                        # Numpy arrays of shape (*,2):
                    'positions_ld'      : Positions of leading dancers,
                    'positions_cb'      : Positions of boundary circles,
                    'positions_all'     : All position values
                }
        }

    
    # Returns instance of State()
    """
    raise NotImplementedError


class StatePreprocessor:
    """Class which stores important meta-variables about the state,
       and provides a learning-algorithm ready vector.
       
    # Init Arguments
        l, w:  Number, number. Length and width of the bridge.
        max_mass:  Number, the maximum mass/area of the bridge.
        allowable_stress:  Number.
        OtherS: To be contemplated.
    
    # Attributes
        l:  Float
        w:  Float
        max_mass:  Float
        allowable_stress:  Float

    # Methods
        preprocess(state): Instance of state to 1d Numpy array
        
    """
    def __init__(self):
        raise NotImplementedError


def _preprocess(l=2.0, w=3.0, max_mass=6.0, allowable_stress=1.0, state=None):
    """Standalone backbone of StatePreprocessor.
    
    # Arguments
        l:  Float
        w:  Float
        max_mass:  Float
        allowable_stress:  Float
        state:  Instance of State()
        
    # Returns 1d Numpy array, preprocessed and ready for use in a neural network.
    """
    raise NotImplementedError


class State:
    """Stores environmental state of our RL agent. A glorified dataclass.
    
    
    # Init Arguments
        # All arguments are an observation of the state of the bridge.
        r_B:  1d numpy array. Radii of leading dancers / boundary circles
        r:  1d numpy array. Radii of all circles. |r| <= |r_B|
        stress:  Number, the stress/sigma of the bridge.
        mass:  Number, the mass of the bridge.
        grad_mass:  1d numpy array, same size as r. Gradient of Mass w.r.t. r.
        points:  2d numpy array, shape (*,2). Represents x,y positions.
        edge_lengths:  1d numpy array of the edge lengths between the points.
        angles:  1d numpy array represnting the angles of all incident edges.
            (Note that this geographic information is unstructured.)
    
    # Attributes
        _r_B, _r, _stress, _mass, _grad_mass, _edge_lengths, _angles, _points:
        All attributes as defined in # Arguments, or None if not initialized.
    
    # Methods
        Getters and _setters for each attribute,
        _set_state(...) to set each attribute at once. Returns self.
        
        To be considered: A .copy() attribute, returning State
    
    # Examples
        # Initialize an empty state and set some basic values
        state = reinforcement_utils.State()
        state._set_mass(300)
        state._set_r(np.array([1,1,1,3]))
        # Same thing, but a one liner
        state = reinforcement_utils.State(mass = 300, r = np.array([1,1,1,3]))
    """
    def __init__(self, r_B          = None,
                       r            = None,
                       stress       = None,
                       mass         = None,
                       grad_mass    = None,
                       edge_lengths = None,
                       angles       = None,
                       points       = None):
        self._r_B           = None
        self._r             = None
        self._stress        = None
        self._mass          = None
        self._grad_mass     = None
        self._edge_lengths  = None
        self._angles        = None
        self._points        = None
        self._set_state(r_B = r_B, r = r, stress = stress, mass = mass,
                        grad_mass = grad_mass, edge_lengths = edge_lengths,
                        angles = angles, points = points)
    
    # Getters and setters
    def get_r_B(self):
        """Getter for 1d np array r_B"""
        return self._r_B
    
    def get_r(self):
        """Getter for 1d np array r"""
        return self._r
    
    def get_stress(self):
        """Getter for number stress"""
        return self._stress
    
    def get_mass(self):
        """Getter number mass"""
        return self._mass
    
    def get_grad_mass(self):
        """Getter for 1d np array grad_mass"""
        return self._grad_mass
    
    def get_edge_lengths(self):
        """Getter for 1d np array edge_lengths"""
        return self._edge_lengths
    
    def get_angles(self):
        """Getter for 1d np array angles"""
        return self._angles
    
    def get_points(self):
        """Getter for 2d np array points of shape (*,2)"""
        return self._points

    def _set_r_B(self,r_B):
        _check_array_and_shape(self._r_B, r_B)
        if self._r is not None:
            if len(self._r) < len(r_B):
                raise ValueError("Can't set r_B where |r_B| > |r|.")
        self._r_B = r_B
    
    def _set_r(self,r):
        _check_array_and_shape(self._r,r)
        if self._r_B is not None:
            if len(self._r_B) > len(r):
                raise ValueError("Can't set r_B where |r_B| > |r|.")
        if self._grad_mass is not None:
            if len(self._grad_mass) > len(r):
                raise ValueError("Can't set r where |r| != |grad_mass|.")
        self._r = r
    
    def _set_stress(self,stress):
        _check_number_and_geqzero(stress)
        self._stress = stress
    
    def _set_mass(self,mass):
        _check_number_and_geqzero(mass)
        self._mass = mass
    
    def _set_grad_mass(self,grad_mass):
        _check_array_and_shape(self._grad_mass, grad_mass)
        if self._r is not None:
            if len(self._r) != len(grad_mass):
                raise ValueError("Can't set grad_mass where |grad_mass| != |r|.")
        self._grad_mass = grad_mass
    
    def _set_edge_lengths(self, edge_lengths):
        _check_array_and_shape(self._edge_lengths,edge_lengths)
        if self._points is not None:
            V = len(self._points)
            E = len(edge_lengths)
            if not V <= E <= 3*V:
                raise ValueError("Can't set points where not |points| <= |edge_lengths| <= 3*|points|")
        self._edge_lengths = edge_lengths
    
    def _set_angles(self, angles):
        _check_array_and_shape(self._angles,angles)
        self._angles = angles
    
    def _set_points(self, points, check_y=True):
        _check_array_and_shape(self._points,points)
        if self._edge_lengths is not None:
            V = len(points)
            E = len(self._edge_lengths)
            if not V <= E <= 3*V:
                raise ValueError("Can't set points where not |points| <= |edge_lengths| <= 3*|points|")
        for point in points:
            if len(point) != 2:
                raise ValueError("Points must be in the real plane! (Set points of shape (n,2))")
            if check_y and point[1] < 0:
                raise ValueError("Can't set a point (x,y) where y < 0!")
        self._points = points
    
    def _set_state(self,
                   r_B          = None,
                   r            = None,
                   stress       = None,
                   mass         = None,
                   grad_mass    = None,
                   edge_lengths = None,
                   angles       = None,
                   points       = None):
        if r_B is not None:          self._r_B       = r_B
        if r is not None:            self._r         = r
        if stress is not None:       self._stress    = stress
        if mass is not None:         self._mass      = mass
        if grad_mass is not None:    self._grad_mass = grad_mass
        if edge_lengths is not None: self._edge_lengths = edge_lengths
        if angles is not None:       self._angles    = angles
        if points is not None:       self._points    = points
        return self
