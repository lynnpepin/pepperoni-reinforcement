""" example.py

An example showing off the work so far, for the data description.

We instantiate a Bridge,
set up an (untrained!) neural network as a RL "policy",
and then complete several policy iterations.

The RL work is not presented here - the policy is basically 'random'.
    No constraints are put into place either.

It is expect that the per-iteration runtime will be dominated by the Mech E code,
    given the network used here will be small. So, despite not performing weight
    updates, we expect the per-iteration runtime here to be representative of the
    final work.
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Activation
from keras.optimizers import Adam

from pepperoni import BridgeHoleDesign
from reinforcement_utils import State, _preprocess, state_from_update
from time import time



""" Instantiating a bridge:
The "BridgeHoleDesign()" class is the face of the M.E. side of this project.
Bridge is an instance of BridgeHoleDesign(), containing all the circle packing
    and FEM goodness needed to do Bridge Stuff. """
    
Bridge = BridgeHoleDesign()



""" Setting up input data:
Bridge provides a function update(), which takes as input a vector "rld_new".

In short, this modifies our bridge per rld_new, and spits out the new bridge state.

More elaborately:
    This vector rld_new is that r_(B) which parameterizes the hole shape.
    Bridge.update() has the side effect of modifying the BridgeHoleDesign, and
                    outputs a dictionary, represnting the new state of the bridge.

This function update() is the most important function here. We show how to use an
    rld vector (here, just taken as the initial value for Bridge) to obtain
    bridge state data. This would modify the bridge if it were a new rld."""

rld = Bridge.rld
data = Bridge.update(rld)



""" Building a simply policy:
We build a simply policy function here, as an untrained model.

First, we need an input vector:
    state_from_update returns an instance of State(). It is the focal point of
    data in from the MechE side to the CompSci side.
The State() class is a glorified dataclass for important state values.

This state is preprocessed into a vector, which we use to construct our network
    for the first time. We use Keras for this. Once constructed, this (untrained)
    model is treated as our "policy"."""
    
state = state_from_update(data)

xin = _preprocess(state)

# Instantiate sequential model with two 128-unit hidden layers
#    and an output the same shape as rld.
# Compare to the "Learning to Optimize" paper which uses a single 50-unit hidden layer.
model = Sequential()
model.add(Dense(128, input_shape = xin.shape, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(len(rld)))

policy = model.predict

""" Iteration:
When the "real" code is finished, here is what a basic loop will look like.
We also do timing and record the resutls below.

Note: The rld vector has the potential to veer into bad shape designs, where the
bridge is very thin and has high stress, or where even the shape of the bridge
is impossible. This is because the policy is not trained/set to avoid this!
"""

n_iterations = 50
lr           = 0.001
rld = np.array(rld)

start_time = time()
times = []
for ii in range(n_iterations):
    # xin has to be reshaped from (454,) to (1,454)
    # rld output, shape (1,*), must be rshaped to shape (*,)
    rld     = rld + lr*policy(xin.reshape(1,-1)).reshape(-1)
    data    = Bridge.update(rld)
    state   = state_from_update(data)
    xin     = _preprocess(state)
    print("== Iteration",str(ii+1),"==")
    print("Mass:   ", data['mass'])
    print("Stress: ", data['sigma'])
    times.append(time())

end_time = time()

tot_time = end_time - start_time

print("Completed", n_iterations, "in", round(tot_time, 3), "seconds.")
print("On average,", round(tot_time/n_iterations,3), "seconds per loop.")

""" Timing:
    Over 50 iterations,
    Completed using standard Pyhon 3.6.3 interpreter,
    Intel Core i7-8750H CPU @ 2.20 GHz

    Average 0.593 seconds per loop with std of .287 seconds.
    # Find change in time from the list "times":
    # t = np.array(times); T = (t[:-1] - t[1:]); np.std(T)

    Expected speedups:
        Using GPU, optimizing code for performance, and using learning that
        converges towards rld values (as poor values require more crunching on
        the FEM side.)
        If we end up using a shallower, narrower network, and less features, we
        expect speedups there too.
    
    Expected slowdowns:
        Accomodating for constraints (i.e. checking that rld does not violate
        hole shape) will require a potentially expensive check in the loop.
        
    "times" list:
    [1552431643.1221504, 1552431643.6072743, 1552431644.0937302, 1552431644.5743804, 1552431645.0409322, 1552431645.5014982, 1552431645.9657779, 1552431646.4401023, 1552431646.9201276, 1552431647.3946123, 1552431647.8612514, 1552431648.3342855, 1552431648.8050015, 1552431649.2772572, 1552431649.7475593, 1552431650.2211962, 1552431650.7046754, 1552431651.1850195, 1552431651.670111, 1552431652.153737, 1552431652.6405816, 1552431653.1376092, 1552431653.6335988, 1552431654.1261098, 1552431654.6216278, 1552431655.1407845, 1552431655.6442273, 1552431656.1494849, 1552431656.6655416, 1552431657.1807716, 1552431657.7052786, 1552431658.2340868, 1552431658.7764468, 1552431659.3133342, 1552431659.8569367, 1552431660.4111168, 1552431660.9756083, 1552431661.5458233, 1552431662.135844, 1552431662.7335024, 1552431663.358539, 1552431663.983555, 1552431664.6340442, 1552431665.3122468, 1552431666.027595, 1552431666.788267, 1552431667.7091064, 1552431668.7976046, 1552431670.0563033, 1552431672.3289075]

"""
