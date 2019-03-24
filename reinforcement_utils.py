"""reinforcement_utils.py

Utilities for reinforcement learning agent.
Provides the point of entry for data from pepperoni into the RL code,
  as well as BridgeState and state preprocessing utility for the RL code.

Provides:
    BridgeState():
        Class that holds relevant state variables for the learning agent.
        Notation is abused. E.g. The agent might use a history of states,
        which would be some list-like of BridgeState().
                
    state_from_update(data):
        The focal point between pepperoni.py and the RL code.
        data is a dict of list, number, or dict, as returned by
          pepperoni.ridgeHoleDesign().update().
        Returns an instance of BridgeState().

    BridgeStatePreprocessor():
        NotImplemented!
        Class which holds preprocessor values, wrapping preprocess_bridge_state.
    
    preprocess_bridge_state():
        NotImplemented!
        Preprocesses a BridgeState() instance and returns a numpy ndarray.
"""


import numpy as np

def state_from_update(data, state = None):
    """Focal point from circlepacking/FEM code to RL side.
    
    Expected to be updated possibly often as the spec for update(rld_new) changes.
    Modifies and returns state if provided, or returns a new state, from the
    update data.
    
    Note: Of all the circles, some subset are boundary circles (cb).
          Of those, a subset are hte leading dances (ld).
    
    # Arguments
        update_data, as provided by pepperoni.BridgeHoleDesign().update(rld).
        I.e. The state of the bridge after updating to the new leading dances rld:
        Currently provided from .update(rld) as a dict:
        {   'raccb'         : List of float; radii of accompanying boundary circles.
            'ri'            : List of float; radii of interior circles.
                            # rld not provided, since that is the input to
                            # update(). The agent should know it itself.
            'sigma'         : Float, representing stress,
            'mass'          : Float, mass of the bridge after update.
            'gmass_r'       : List of float; gradient w.r.t. r of mass,
            'gmass_rld'     : List of float, 
                            # gmass_rld are those values of gmass_r which
                            # correspond to indices of leading dancers.
            'geometry_info' : Dictionary, of format:
                                        # Lists of float:
                {   'angles_ld'         : Surround angles of leading dancers,
                    'angles_accb'       : Surround angles of accompanying boundary circles,
                                        # Float:
                    'total_length_ld'   : Total length of leading edges,
                    'total_length_accb' : Total edge length of boundary,
                                        # List of float:
                    'edges_ld'          : Edge lengths of leading dancers,
                    'edges_accb'        : Edge lengths of accompanying boundary circles,
                                        # Numpy arrays of shape (*,2):
                    'positions_ld'      : Positions of leading dancers,
                    'positions_accb'    : Positions of accompanying boundary circles,
                    'positions_ci'      : Positions of the interior points
                }
        }
        
        state (optional): An instance of state to update. If not set, a new
            instance is created and reutrned.

    # Returns an instance of BridgeState() with its attributes set/updated.
    """
    # Note that rld is not provided, and so rld is not set/updated by this.
    if state is None:
        state = BridgeState()
        # an instance of state is provided; modify that instance, and return it.
    state.set_state(raccb              = np.array(data['raccb']),
                     ri                 = np.array(data['ri']),
                     stress             =          data['sigma'],
                     mass               =          data['mass'],
                     grad_mass          = np.array(data['gmass_r']),
                     grad_mass_ld       = np.array(data['gmass_rld']),
                     edge_lengths_ld    = np.array(data['geometry_info']['edges_ld']),
                     edge_lengths_accb  = np.array(data['geometry_info']['edges_accb']),
                     angles_ld          = np.array(data['geometry_info']['angles_ld']),
                     angles_accb        = np.array(data['geometry_info']['angles_accb']),
                     # positions are provided as numpy arrays for some reason
                     points_ld          =          data['geometry_info']['positions_ld'],
                     points_accb        =          data['geometry_info']['positions_accb'],
                     points_ci          =          data['geometry_info']['positions_ci'])
    return state



####
# Preprocessing stuff
####
class _BridgeStatePreprocessor:
    """Class which stores important meta-variables about the state,
       and provides a learning-algorithm ready vector.
       
    # Init Arguments
        l, w:  Number, number. Length and width of the bridge.
        max_mass:  Number, the maximum mass/area of the bridge.
        allowable_stress:  Number.
        Others: To be contemplated.
    
    # Attributes
        l:  Float
        w:  Float
        max_mass:  Float
        allowable_stress:  Float

    # Methods
        preprocess(state): Instance of state to 1d Numpy array
    
    # Not yet implemented, but may be useful in the future. Hmm...
    """
    def __init__(self):
        raise NotImplementedError


def _normalize_01(x, b=1.0, a = 0.0):
    """Normalize x to [0,1] bounds, given max value b and min value a."""
    # todo: Consider generating warnings if x > b or x < a, or if b <= a
    # todo: If no b, a, use the max, min of x.
    return (x-a)/(b-a)


def _normalize_angle(x, rad=True):
    """Given an angle, return the (cos(x), sin(x)) pair represnting it.
    Given an numpy array of shape (a,b,...,c), returns an array of shape
    (a,b,...,c,2), where the last value indexes cos or sin."""
    if not rad:
        x = x*np.pi/180
    
    return np.moveaxis(np.array([np.cos(x), np.sin(x)]), 0, -1)


def preprocess_bridge_state(state = None, l=20.0, w=10.0, max_mass=400.0, allowable_stress=200.0):
    """Standalone backbone of BridgeStatePreprocessor.
    
    # Arguments
        l:  Float
        w:  Float
        max_mass:  Float
        allowable_stress:  Float
        state:  Instance of BridgeState()
        
    # Returns 1d Numpy array, preprocessed and ready for use in a neural network.
    """
    length = np.sqrt(l**2 + w**2)
    out = list()
    
    rld = state.get_rld()
    raccb = state.get_raccb()
    ri = state.get_ri()
    stress = state.get_stress()
    mass = state.get_mass()
    grad_mass = state.get_grad_mass()
    grad_mass_ld = state.get_grad_mass_ld()
    edge_lengths_ld = state.get_edge_lengths_ld()
    edge_lengths_accb = state.get_edge_lengths_accb()
    angles_ld = state.get_angles_ld()
    angles_accb = state.get_angles_accb()
    points_ld = state.get_points_ld()
    points_accb = state.get_points_accb()
    points_ci = state.get_points_ci()
    
    euclidean_attrs = [rld, raccb, ri, edge_lengths_ld, edge_lengths_accb,
                       points_ld, points_accb, points_ci]

    for attr in euclidean_attrs:
        if attr is not None:
            attr = (_normalize_01(attr, b = length))
    
    for attr in [angles_ld, angles_accb]:
        if attr is not None:
            attr = np.array(_normalize_angle(attr))

    if stress is not None:
        stress = np.array(_normalize_01(stress, b = allowable_stress))
    
    if mass is not None:
        mass = np.array(_normalize_01(mass, b = max_mass))

    attributes = [rld, raccb, ri, stress, mass, grad_mass, grad_mass_ld,
                  edge_lengths_ld, edge_lengths_accb,
                  angles_ld, angles_accb, points_ld, points_accb, points_ci]

    for attr in attributes:
        if attr is not None:
            out.append(attr.ravel())

    return np.concatenate(out)



####
# State stuff
####
class BridgeState:
    """Stores environmental state of our RL agent. A glorified dataclass.
    
    
    # Init Arguments
        # All arguments are an observation of the state of the bridge.
        # Suffixes: _ld refer to the leading dancers on the border,
        #           _accb refers to accompanying boarder points,
        #           _ci refers to all other interior points.
        rld, raccb, ri:
            1d numpy arrays, radii of circles.
            rld is generated by the agent's policy.
        stress:  Number, the stress/sigma of the bridge.
        mass:  Number, the mass of the bridge.
        grad_mass: 
            1d numpy array, same size as r. Gradient of Mass w.r.t. r.
        grad_mass_ld:
            1d numpy array. Those elements of grad_mass corresponding to leading dancers.
        points_ld, points_accb, points_ci:
            2d numpy arrays, shape (*,2). Represents x,y positions.
        edge_lengths_ld, edge_lengths_accb: 
            1d numpy arrays of the edge lengths between the points.
        angles_ld, angles_accb:
            1d numpy arrays represnting the angles of all incident edges.
            (Note that this geographic information is unstructured.)
    
    # Attributes
        #All attributes as defined in # Arguments, or None if not initialized.
        _rld, _raccb, _ri,
        _stress, _mass, _grad_mass, _grad_mass_ld,
        _points_ld, _points_accb, _points_ci,
        _edge_lengths_ld, _edge_lengths_accb, _angles_ld, _angles_accb,
        
    
    # Methods
        Getters and setters for each attribute,
        set_state(...) to set each attribute at once. Returns self.
        
        To be considered: A .copy() attribute, returning a new BridgeState instance.
    
    # Examples
        # Initialize an empty state and set some basic values
        state = reinforcement_utils.BridgeState()
        state.set_mass(300)
        state.set_ri(np.array([1,1,1,3]))
        # Same thing, but a one liner
        state = reinforcement_utils.BridgeState(mass = 300, ri = np.array([1,1,1,3]))
    """
    def __init__(self, rld              = None,
                       raccb            = None,
                       ri               = None,
                       stress           = None,
                       mass             = None,
                       grad_mass        = None,
                       grad_mass_ld     = None,
                       edge_lengths_ld  = None,
                       edge_lengths_accb =None,
                       angles_ld        = None,
                       angles_accb      = None,
                       points_ld        = None,
                       points_accb      = None,
                       points_ci        = None):
        # Instantiate all variables.
        # a, b, c = [None]*3 is short hand for a = None; b = None; c = None;
        self._rld, self._raccb, self._ri = [None]*3
        self._stress, self._mass = [None]*2
        self._grad_mass, self._grad_mass_ld = [None]*2
        self._edge_lengths_ld, self._edge_lengths_accb = [None]*2
        self._angles_ld, self._angles_accb = [None]*2
        self._points_ld, self._points_accb, self._points_ci = [None]*3
         
        self.set_state(rld = rld, raccb = raccb, ri = ri,
                        stress = stress, mass = mass,
                        grad_mass = grad_mass, grad_mass_ld = grad_mass_ld,
                        edge_lengths_ld  = edge_lengths_ld,
                        edge_lengths_accb = edge_lengths_accb,
                        angles_ld = angles_ld, angles_accb = angles_accb,
                        points_ld = points_ld, points_accb = points_accb, points_ci = points_ci)
    
    # Getters and setters
    def get_rld(self):
        """Getter for 1d positive numpy array rld"""
        return self._rld

    def get_raccb(self):
        """Getter for 1d positive numpy array raccb"""
        return self._raccb

    def get_ri(self):
        """Getter for 1d positive numpy array ri"""
        return self._ri

    def get_stress(self):
        """Getter for positive number stress"""
        return self._stress

    def get_mass(self):
        """Getter for positive number mass"""
        return self._mass

    def get_grad_mass(self):
        """Getter for 1d numpy array grad_mass"""
        return self._grad_mass

    def get_grad_mass_ld(self):
        """Getter for 1d numpy array grad_mass_ld"""
        return self._grad_mass_ld

    def get_edge_lengths_ld(self):
        """Getter for 1d positive numpy array edge_lengths_ld"""
        return self._edge_lengths_ld

    def get_edge_lengths_accb(self):
        """Getter for 1d positive numpy array edge_lengths_accb"""
        return self._edge_lengths_accb

    def get_angles_ld(self):
        """Getter for 1d numpy array angles_ld"""
        return self._angles_ld

    def get_angles_accb(self):
        """Getter for 1d numpy array angles_accb"""
        return self._angles_accb

    def get_points_ld(self):
        """Getter for 2d numpy array points_ld"""
        return self._points_ld

    def get_points_accb(self):
        """Getter for 2d numpy array points_accb of shape (*,2)"""
        return self._points_accb

    def get_points_ci(self):
        """Getter for 2d numpy array points_ci of shape (*,2)"""
        return self._points_ci
    
    
    def set_rld(self,rld):
        _check_array_and_shape(self.get_rld(), rld)
        self._rld = rld

    def set_raccb(self,raccb):
        _check_array_and_shape(self.get_raccb(), raccb)
        self._raccb = raccb

    def set_ri(self,ri):
        _check_array_and_shape(self.get_ri(), ri)
        self._ri = ri

    def set_stress(self,stress):
        _check_number_and_geqzero(stress)
        self._stress = stress

    def set_mass(self,mass):
        _check_number_and_geqzero(mass)
        self._mass = mass

    def set_grad_mass(self,grad_mass):
        _check_array_and_shape(self.get_grad_mass(), grad_mass)
        if self.get_rld() is not None and \
           self.get_raccb() is not None and \
           self.get_ri() is not None:
            size_r = len(self.get_rld()) + len(self.get_raccb()) + len(self.get_ri())
            if size_r != len(grad_mass):
                raise ValueError("Can't set grad_mass where |grad_mass| != |r_all|.")
        self._grad_mass = grad_mass

    def set_grad_mass_ld(self,grad_mass_ld):
        if self.get_rld() is not None:
            if len(self.get_rld()) != len(grad_mass_ld):
                raise ValueError("Can't set grad_mass where |grad_mass| != |rld|.")
        _check_array_and_shape(self.get_grad_mass_ld(), grad_mass_ld)
        self._grad_mass_ld = grad_mass_ld

    def set_edge_lengths_ld(self,edge_lengths_ld):
        _check_array_and_shape(self.get_edge_lengths_ld(), edge_lengths_ld)
        self._edge_lengths_ld =  edge_lengths_ld

    def set_edge_lengths_accb(self,edge_lengths_accb):
        _check_array_and_shape(self.get_edge_lengths_accb(), edge_lengths_accb)
        self._edge_lengths_accb = edge_lengths_accb

    def set_angles_ld(self,angles_ld):
        _check_array_and_shape(self.get_angles_ld(), angles_ld)
        self._angles_ld = angles_ld

    def set_angles_accb(self,angles_accb):
        _check_array_and_shape(self.get_angles_accb(), angles_accb)
        self._angles_accb = angles_accb

    def set_points_ld(self,points_ld):
        _check_array_and_shape(self.get_points_ld(), points_ld, one_d = False)
        _check_points(points_ld)
        self._points_ld = points_ld

    def set_points_accb(self,points_accb):
        _check_array_and_shape(self.get_points_accb(), points_accb, one_d = False)
        _check_points(points_accb)
        self._points_accb = points_accb

    def set_points_ci(self,points_ci):
        _check_array_and_shape(self.get_points_ci(), points_ci, one_d = False)
        _check_points(points_ci)
        self._points_ci = points_ci

    
    def set_state(self, rld                = None,
                         raccb              = None,
                         ri                 = None,
                         stress             = None,
                         mass               = None,
                         grad_mass          = None,
                         grad_mass_ld       = None,
                         edge_lengths_ld    = None,
                         edge_lengths_accb  = None,
                         angles_ld          = None,
                         angles_accb        = None,
                         points_ld          = None,
                         points_accb        = None,
                         points_ci          = None):
        if rld is not None:
            self.set_rld(rld)

        if raccb is not None:
            self.set_raccb(raccb)

        if ri is not None:
            self.set_ri(ri)

        if stress is not None:
            self.set_stress(stress)

        if mass is not None:
            self.set_mass(mass)

        if grad_mass is not None:
            self.set_grad_mass(grad_mass)

        if grad_mass_ld is not None:
            self.set_grad_mass_ld(grad_mass_ld)

        if edge_lengths_ld is not None:
            self.set_edge_lengths_ld(edge_lengths_ld)

        if edge_lengths_accb is not None:
            self.set_edge_lengths_accb(edge_lengths_accb)

        if angles_ld is not None:
            self.set_angles_ld(angles_ld)

        if angles_accb is not None:
            self.set_angles_accb(angles_accb)

        if points_ld is not None:
            self.set_points_ld(points_ld)

        if points_accb is not None:
            self.set_points_accb(points_accb)

        if points_ci is not None:
            self.set_points_ci(points_ci)

        return self


# Misc functions used in BridgeState()
def _check_array_and_shape(x, new_x, one_d=True):
    """Check to ensure new_x as an array is the same shape as x.
    if 1d=True, make sure the shape of new_x is (*,)"""
    if not isinstance(new_x, np.ndarray):
        raise TypeError("Input must be a numpy array!")
    if isinstance(x, np.ndarray):
        # if x is an array, make sure
        # x and new_x have matching shape.  
        if not(x.shape == new_x.shape):
            raise ValueError("Shape mismatch setting new value.")
    if one_d and isinstance(new_x, np.ndarray) and len(new_x.shape) != 1:
        raise TypeError("Must be a 1d array.")


def _check_number_and_geqzero(x):
    """Ensure x is a nubmer >= 0."""
    if not isinstance(x, (int, float)):
        raise TypeError("Not a number!")
    elif not (x > 0):
        raise ValueError("Not greater than zero!")


def _check_points(points):
    """Check that np array points is of shape (*2) and of proper values"""
    if points.shape[1] != 2:
        raise ValueError("Points must be in the real plane! (Set points of shape (n,2))")    
    for point in points:
        if point[1] < 0:
            raise ValueError("Points y must be greater than 0!")

