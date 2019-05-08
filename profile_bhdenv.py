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
        random_action = .001*(.5-np.random.rand(input_shape))
        ob, reward, done, info = env.step(random_action)
        if done:
            break

'''
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    457/1    0.002    0.000   45.995   45.995 {built-in method builtins.exec}
        1    0.001    0.001   45.995   45.995 profile_bhdenv.py:5(<module>)
      105    0.017    0.000   39.601    0.377 /home/lynn/proj/pepperoni-reinforcement/pepperoni.py:210(_finite_element_analysis)
      105   35.899    0.342   39.549    0.377 /home/lynn/proj/pepperoni-reinforcement/_FEM.py:129(_FEM)
       95    0.002    0.000   36.850    0.388 /home/lynn/proj/pepperoni-reinforcement/pepperoni.py:100(update)
       75    0.001    0.000   28.663    0.382 /home/lynn/proj/pepperoni-reinforcement/gym_wrappers.py:187(step)
       10    0.000    0.000   17.029    1.703 /home/lynn/proj/pepperoni-reinforcement/gym_wrappers.py:249(reset)
       10    0.000    0.000   13.226    1.323 /home/lynn/proj/pepperoni-reinforcement/pepperoni.py:11(__init__)
        5    0.000    0.000    8.508    1.702 /home/lynn/proj/pepperoni-reinforcement/gym_wrappers.py:153(__init__)
      105    0.715    0.007    5.415    0.052 /home/lynn/proj/pepperoni-reinforcement/pepperoni.py:789(_calculate_radii)
       10    0.003    0.000    5.049    0.505 /home/lynn/proj/pepperoni-reinforcement/pepperoni.py:955(_generate_circlepacking)
  1126225    4.146    0.000    4.600    0.000 /home/lynn/proj/pepperoni-reinforcement/pepperoni.py:524(_theta_arround)
    21000    1.626    0.000    2.607    0.000 /home/lynn/proj/pepperoni-reinforcement/_FEM.py:57(_membershiptest)
   567945    0.981    0.000    0.981    0.000 /home/lynn/proj/pepperoni-reinforcement/_FEM.py:21(_ccw)
       95    0.002    0.000    0.740    0.008 /home/lynn/proj/pepperoni-reinforcement/pepperoni.py:1057(_modify_circlepacking)
      105    0.000    0.000    0.691    0.007 <decorator-gen-7>:1(cg)
'''
