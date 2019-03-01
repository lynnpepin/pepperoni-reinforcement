# -*- coding: utf-8 -*-
from pepperoni import *
from FEM import *

l = 20.0 # half length of the bridge
h = 10.0 # height of the bridge
a_ell = 16.0 # the initial shape of the hole is an ellipse
b_ell = 8.0
delta = 2.0 # distance between the points of triangulation
eps = 0.01 # error tolerate for circle packing calculation
delta_r = 0.001 # the changes in each iteration in the process of calculation

# check the plot  
tri, x_domain, y_domain = generate_triangulation(l, h, a_ell, b_ell, delta)


LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin = generate_circlepacking(tri, 
                                    x_domain, y_domain, delta, eps, delta_r)
for r in circles:
    print('Expected angle: ',r.totall_angle*180/np.pi, 
          'Actual angle: ',theta_arround(r)*180/np.pi)
# It is not easy to calculate the right circle packing by hand, 
# we might check the surround angle in stead.
    
test_theta = theta_arround(circles[0])

yesorno1 = in_circles(circles[0],LD)
yesorno2 = in_circles(circles[0],AD)

dr = [0]*len(LD)
for i in range(0,len(LD)):
    dr[i] = -0.2 + 0.4*random.uniform(0,1)
modify_circlepacking(dr,LD,AD)
print('After Modification')
for r in AD:
    print('Expected angle: ',r.totall_angle*180/np.pi, 
          'Actual angle: ',theta_arround(r)*180/np.pi)
    
edges, r_ld, r = generate_boundary_edges(LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin)

sigma, area = FEM(edges)