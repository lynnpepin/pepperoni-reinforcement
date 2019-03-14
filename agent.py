import numpy as np
from pepperoni import BridgeHoleDesign as BHD
from reinforcement_utils import State, state_from_update, _preprocess

class BasicAgent:
    def __init__(self):
        raise NotImplementedError
    
    """A bare-bones agent, providing the structure for smarter agents.

    # Init Arguments
        # Attribute arguments:
            state:  Instance of State. Optional.
                Provides the initial state of the bridge.
            rld:  Instance of numpy.ndarray. Optional.
                Leading dancers radii in the circle packing.
                Overrides state._rld is provided.
            environment:  Instance of BHD, providing update().
                Update maps rld to a dictionary, representing bridge state.
            preprocess:  Function : State() --> numpy.ndarray
                Preprocesses a bridge state for a neural network. 
            policy:  Function : numpy.ndarray --> numpy.ndarray
                Maps preprocessed state array to a new choice of rld.
                        
    # Attributes    
        # Attribute arumgents
        # Other attributes:
            _raw_update:  Instance of dict
                The unprocessed dictionary provided by environment.update()
            
    # Methods
        Getters and setters for each attribute.
        _step_1_send_rld:
            Calls environment.update(self.rld), sets self._raw_update
        _step_2_process_update:
            From self._raw_update using state_from_update, set self.state
        _step_3_call_policy_and_set_rld:
            Set rld = self.policy(self.preprocess(self.state))
        iterate():
            Call steps 1 through 3 in order
            
    # Examples
    
    # Extension plans
        1. Recording history, and giving the agent the ability to choose to roll-back.
        2. A dumb, hard-coded agent that explores per simplex.
        3. A dumb, hard-coded agent that explores per GD.
    """


