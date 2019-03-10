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

class State():
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
        return self._r_B
    
    def get_r(self):
        return self._r
    
    def get_stress(self):
        return self._stress
    
    def get_mass(self):
        return self._mass
    
    def get_grad_mass(self):
        return self._grad_mass
    
    def get_edge_lengths(self):
        return self._edge_lengths
    
    def get_angles(self):
        return self._angles
    
    def get_points(self):
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
