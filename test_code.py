# -*- coding: utf-8 -*-
from pepperoni import *
from FEM import *


delta = 1.0 # distance between the points of triangulation
eps = 0.01 # error tolerate for circle packing calculation
delta_r = 0.001 # the changes in each iteration in the process of calculation

# A simple example
x_tri = [0, 1, 0];
y_tri = [0, 0, 1];
points_tri = np.vstack([x_tri,y_tri]).T 
tri1 = Delaunay(points_tri)
LD1, AD1, circles1, faces1, cb1, anchor_x1, anchor_y1, cb_origin1 = generate_circlepacking(tri1, 
                                     0.5, 0.1*eps, 0.1*delta_r)
print('radii of circles1')
print('r1 ', 'Exact Values ', 1-1/np.sqrt(2), ' Calculated ', circles1[0].radius )
print('r2 ', 'Exact Values ', 1/np.sqrt(2), ' Calculated ', circles1[1].radius )
# circles1[0].radius == r1 ~ 0.29289321881345254
# circles1[1].rdius == 1/sqrt(2)~0.7071067811865475
# circles1[2].radius == 1/sqrt(2)~0.7071067811865475

# A bridge example
l = 20.0 # half length of the bridge
h = 10.0 # height of the bridge
a_ell = 16.0 # the initial shape of the hole is an ellipse
b_ell = 8.0
tri= generate_triangulation(l, h, a_ell, b_ell, delta)


LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin = generate_circlepacking(tri, 
                                     delta, eps, delta_r)

# It is not easy to calculate the right circle packing by hand, 
# we might check the surround angle in stead.
    
test_theta = theta_arround(circles[0])

yesorno1 = in_circles(circles[0],LD)
yesorno2 = in_circles(circles[0],AD)

dr = [0]*len(LD)
for i in range(0,len(LD)):
    dr[i] = -0.2 + 0.4*random.uniform(0,1)
#modify_circlepacking(dr,LD, AD, 0.1*eps, 0.1*delta_r)
    
draw_circles(circles,l)
    
edges, r_ld, r = generate_boundary_edges(LD, circles, faces, anchor_x, anchor_y, cb_origin)

sigma, area = FEM(edges)
print('Sigma = ', sigma, 'Area = ', area)