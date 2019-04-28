# To use:
# python3 -m cProfile -o profile_outpiut.txt profile_bhdenv.py
# import pstats; p = pstats.Stats('profile_output.txt'); p.sort_stats('cumulative').print_stats(40)

import numpy as np
from gym_wrappers import BHDEnv

iterations = 5
updates_per_iteration = 15

for ii in range(iterations):
    print("Iteration:", ii)
    env = BHDEnv(length = 20.0, height = 10.0, allowable_stress = 200.0)
    ob = env.reset()
    
    input_shape = env.action_space.shape[0]
    
    for jj in range(updates_per_iteration):
        print("  Step:", jj)
        random_action = .001*np.random.rand(input_shape)
        ob, reward, done, info = env.step(random_action)
        if done:
            break


