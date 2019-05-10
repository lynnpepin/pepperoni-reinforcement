from pepperoni import BridgeHoleDesign as BHD
from config_dict import CONFIG
import numpy as np


allowable_stress = CONFIG['allowable_stress']
# Known good starting lr
lr = 2**-4
decay = .5

RENDER      = False
bridge      = BHD()
rld         = np.array(bridge.rld)
data        = bridge.update(rld)
stress      = data['sigma']
gmass_rld   = np.array(data['gmass_rld'])


masses = []
sigmas = []
rlds = []
print("","Iter","LR","mass","sigma", sep="\t")
def print_values(label, ii, lr, mass, sigma):
    print(label, ii, "2**"+str(int(np.log2(lr))), round(mass,3),
    np.format_float_scientific(sigma,2), sep="\t")

for ii in range(21):
    # Quickly get a new rld that isn't too close to 0
    new_rld = rld - lr*gmass_rld
    while np.any(new_rld <= 2**-16):
        lr = lr * decay
        new_rld = rld - lr*gmass_rld
    
    # Make a brand new bridge, update with this new rld
    bridge = BHD()
    new_data = bridge.update(new_rld)
    
    
    ## Check to see if the new rld provides a legal bridge design
    if new_data['sigma'] >= allowable_stress:
        print_values("Bad!!", ii, lr, new_data['mass'], new_data['sigma'])
        lr = lr * decay
        #print_values("Reset", ii, lr, data['mass'], data['sigma'])
    
    # new_rld is good; define it as our rld and continue
    else:
        rld = new_rld
        data = new_data
        print_values("Good!", ii, lr, data['mass'], data['sigma'])
    
    if RENDER:
        bridge.render()
    
    masses.append(data['mass'])
    sigmas.append(data['sigma'])
    rlds.append(rld)
