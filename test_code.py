# -*- coding: utf-8 -*-
from pepperoni import *

LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin = generate_circlepacking()

test_theta = theta_arround(circles[0])

yesorno1 = in_circles(circles[0],LD)
yesorno2 = in_circles(circles[0],AD)

dr = [0]*len(LD)
for i in range(0,len(LD)):
    dr[i] = -0.1 + 0.2*random.uniform(0,1)
modify_circlepacking(dr,LD,AD)

edges, r_ld, r = generate_boundary_edges(LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin)