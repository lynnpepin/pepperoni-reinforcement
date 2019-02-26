# Pepperoni module
"""
Data Structures: CircleVertex, HalfEdge, Face
Functions: generate_circlepacking(), generate_boundary_edges()
Usage: use edges, r_ld, r = generate_boundary_edges() to get the coordinates of the points
on boundary, radii of leading dancers and radii of all circles 
"""
import math
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
class CircleVertex:
    incident_halfedge = []
    neighbors = []
    placed = 0
    def __init__(self, i=[], r=[], ang=[]):
        self.index = i
        self.radius = r
        self.totall_angle = ang

class HalfEdge:
    flip = []
    next = []
    prev = []

    def __init__(self, vi=[], vj=[], f=[]):
        self.source = vi
        self.target = vj
        self.face_index = f

class Face:
    def __init__(self, v1=[], v2=[], v3=[], i=[]):
        self.index = i
        self.vertex1 = v1
        self.vertex2 = v2
        self.vertex3 = v3
        self.halfedge1 = HalfEdge(v1, v2, i)
        self.halfedge2 = HalfEdge(v2, v3, i)
        self.halfedge3 = HalfEdge(v3, v1, i)
        self.halfedge1.next=self.halfedge2;
        self.halfedge2.next=self.halfedge3;
        self.halfedge3.next=self.halfedge1;
        if self.vertex1.incident_halfedge == []:
            self.vertex1.incident_halfedge = self.halfedge1
        if self.vertex2.incident_halfedge == []:
            self.vertex2.incident_halfedge = self.halfedge2
        if self.vertex3.incident_halfedge == []:
            self.vertex3.incident_halfedge = self.halfedge3


def in_circles(c,Cir):
    """
    determine whether c in circle list
    """
    for ci in Cir:
        if c.index == ci.index:
            yesorno = True
            break
    else:
        yesorno = False
    return yesorno


def theta_arround(cv):
    """
    calculate the surround angle of a circle vertex cv
    """
    r = cv.radius
    theta = 0
    for i in range(0,len(cv.neighbors)-1 ):
        rj = cv.neighbors[i].radius
        rk = cv.neighbors[i+1].radius
        theta = math.acos( ( (r+rj)**2 + (r+rk)**2 - 
        (rj+rk)**2 )/( 2*(r+rj)*(r+rk) ) ) + theta
    return theta

def generate_circlepacking():
    l = 20.0
    h = 10.0
    a_ell = 16.0
    b_ell = 8.0
    delta = 2.0
    eps = 0.01
    delta_r = 0.001
    x_domain = []
    y_domain = []
    x = np.linspace(0.0, l, math.ceil(l/delta))
    y = np.linspace(0.0, h, math.ceil(h/delta))
    for xi in x:
        for yi in y:
            if (xi/a_ell)**2+(yi/b_ell)**2 < 1:
                x_domain.append(xi)
                y_domain.append(yi)
    points = np.vstack([x_domain,y_domain]).T
    tri = Delaunay(points)
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy())
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
    circles = []
    n_c = len(x_domain)
    for i in range(0, n_c):
        circles.append(CircleVertex())
    
    for i in range(0, n_c):
        if x_domain[i] == 0 and y_domain[i] > 0:
            circles[i].index = i
            circles[i].radius = delta/2
            circles[i].totall_angle = np.pi
            circles[i].x = x_domain[i]
            circles[i].y = y_domain[i]
        elif x_domain[i] > 0 and y_domain[i] == 0:
            circles[i].index = i
            circles[i].radius = delta/2
            circles[i].totall_angle = np.pi
            circles[i].x = x_domain[i]
            circles[i].y = y_domain[i]
        elif x_domain[i] == 0 and y_domain[i] == 0:
            circles[i].index = i
            circles[i].radius = delta/2
            circles[i].totall_angle = np.pi/2
            circles[i].x = x_domain[i]
            circles[i].y = y_domain[i]
        else:
            circles[i].index = i
            circles[i].radius = delta/2
            circles[i].totall_angle = 0
            circles[i].x = x_domain[i]
            circles[i].y = y_domain[i]
    faces = []
    halfedges = []
    n_t = len(tri.simplices)
    for i in range(0, n_t):
        faces.append(Face(circles[tri.simplices[i][0]], circles[tri.simplices[i][1]], 
        circles[tri.simplices[i][2]], i))
        halfedges.append(faces[i].halfedge1)
        halfedges.append(faces[i].halfedge2)
        halfedges.append(faces[i].halfedge3)
    for j in range(0, len(halfedges)):
        for i in range(0, len(halfedges)):
            if halfedges[i].source.index == halfedges[j].target.index and \
               halfedges[i].target.index == halfedges[j].source.index:
               halfedges[j].flip=halfedges[i]
    cb = []
    for edge in halfedges:
        if edge.flip == []:
            cb.append(edge.source)
            cb.append(edge.target)
            traveller = edge
            break
    while cb[-1].index != cb[0].index:
        while traveller.next.flip != []:
            traveller=traveller.next.flip
        traveller=traveller.next
        cb.append(traveller.target)
    ci = []
    for c in circles:
        if in_circles(c, cb) == False:
            c.totall_angle = 2*np.pi
            ci.append(c)
    #find neighbor for every interior circles
    for i in range(0,len(ci)):
        traveller2 = ci[i].incident_halfedge
        ci[i].neighbors = [traveller2.target]
        traveller2 = traveller2.next.next.flip
        ci[i].neighbors.append(traveller2.target)
        while ci[i].neighbors[-1].index != ci[i].neighbors[0].index:
            traveller2 = traveller2.next.next.flip
            ci[i].neighbors.append(traveller2.target)
    #find neighbor for every boundary circles
    for c in cb:
        traveller3 = c.incident_halfedge
        if traveller3.flip == []:
            c.neighbors = [traveller3.target]
            c.neighbors.append(traveller3.next.target)
            while traveller3.next.next.flip != []:
                traveller3 = traveller3.next.next.flip
                c.neighbors.append(traveller3.next.target)
        else:
            while traveller3.flip != []:
                traveller3 = traveller3.flip.next
            c.neighbors = [traveller3.target]
            c.neighbors.append(traveller3.next.target)
            while traveller3.next.next.flip != []:
                traveller3 = traveller3.next.next.flip
                c.neighbors.append(traveller3.next.target)
    #set surround angle for boundary undetermined circles
    #find the start point and the end point
    x_max = np.max(x_domain)
    y_max = np.max(y_domain)
    for i in range(0,len(cb)):
        if cb[i].x == x_max and cb[i].y == 0:
            cb_start = cb[i]
            cb_start_index = i
        elif cb[i].x == 0 and cb[i].y == y_max:
            cb_end = cb[i]
            cb_end_index = i
        elif cb[i].x == 0 and cb[i].y == 0:
            cb_origin = cb[i]
            cb_origin_index = i
    #collect the leading dancers
    j = cb_start_index
    LD = []
    while j != cb_end_index:
        LD.append(cb[j])
        j = j+1
        if j == len(cb):
            j = 1
    LD.append(cb_end)
    #collect the accompanying dancers
    AD = []
    for c in circles:
        if in_circles(c, LD) == False:
            AD.append(c)
    #determine the surround angle for leading dancers
    for j in range(0, len(LD)):
        surround_angle = 0
        for i in range(0, len(LD[j].neighbors)-1):
            L1 = np.sqrt( ( LD[j].x - LD[j].neighbors[i].x )**2 + ( LD[j].y - 
            LD[j].neighbors[i].y )**2 )
            L2 = np.sqrt( (LD[j].x - LD[j].neighbors[i+1].x)**2 + (LD[j].y - 
            LD[j].neighbors[i+1].y)**2 )
            L3 = np.sqrt( (LD[j].neighbors[i].x - LD[j].neighbors[i+1].x)**2 + 
            (LD[j].neighbors[i].y - LD[j].neighbors[i+1].y)**2 )
            alpha = math.acos( (L1**2+L2**2-L3**2)/(2*L1*L2) )
            surround_angle = surround_angle + alpha
        LD[j].totall_angle = surround_angle
    #Calculate radii for interior and undertermined circles
    theta_diff = [0]*n_c
    for i in range(0,n_c):
        if circles[i].index != LD[1].index:
            theta_diff[i] = theta_arround(circles[i]) - circles[i].totall_angle
    while np.max(theta_diff) > eps or np.min(theta_diff) < -eps:
        for i in range(0, n_c):
            if theta_diff[i] < 0:
                if circles[i].index != LD[1].index:
                    circles[i].radius = circles[i].radius - delta_r*circles[i].radius
            elif theta_diff[i] > 0:
                if circles[i].index != LD[1].index:
                    circles[i].radius = circles[i].radius + delta_r*circles[i].radius
        for i in range(0, n_c):
            if circles[i].index != LD[1].index:
                theta_diff[i] = theta_arround(circles[i]) - circles[i].totall_angle
    #Layout circles
    faces_copy = faces.copy()
    #anchers lie along x axis
    i = cb_origin_index
    anchor_x = []
    while cb[i].y == 0:
        anchor_x.append(cb[i])
        i = i+1
        if i == len(cb):
            i = 1
    #anchers lie along y axis
    i = cb_origin_index-1
    if i < 0:
        i = len(cb)-2
    anchor_y = []
    while cb[i].x == 0:
        anchor_y.append(cb[i])
        i = i-1
        if i < 0:
            i = len(cb) - 2
    #layout anchers along x
    x_coord = 0
    anchor_x[0].x = 0
    anchor_x[0].y = 0
    anchor_x[0].placed = 1      
    for i in range(1, len(anchor_x)):
        x_coord = x_coord + anchor_x[i-1].radius + anchor_x[i].radius
        anchor_x[i].x = x_coord
        anchor_x[i].y = 0
        anchor_x[i].placed = 1
    #layout anchers along y
    y_coord = 0
    y_coord = y_coord + cb_origin.radius + anchor_y[0].radius
    anchor_y[0].y = y_coord;
    anchor_y[0].x = 0;
    anchor_y[0].placed = 1;
    for i in range(1, len(anchor_y)):
        y_coord = y_coord + anchor_y[i-1].radius + anchor_y[i].radius;
        anchor_y[i].y = y_coord;
        anchor_y[i].x = 0;
        anchor_y[i].placed = 1;
    #layour other circles
    i_face=0
    while faces_copy != []:
        if faces_copy[i_face].vertex1.placed + faces_copy[i_face].vertex2.placed + \
        faces_copy[i_face].vertex3.placed == 3:
            faces_copy.pop(i_face)
            i_face = 0
        elif faces_copy[i_face].vertex1.placed + faces_copy[i_face].vertex2.placed + \
        faces_copy[i_face].vertex3.placed == 2:
            if faces_copy[i_face].vertex1.placed == 1 and \
            faces_copy[i_face].vertex2.placed == 1:
                vi = faces_copy[i_face].vertex1
                vj = faces_copy[i_face].vertex2
                vk = faces_copy[i_face].vertex3
            elif faces_copy[i_face].vertex2.placed == 1 and \
            faces_copy[i_face].vertex3.placed == 1:
                vi = faces_copy[i_face].vertex2
                vj = faces_copy[i_face].vertex3
                vk = faces_copy[i_face].vertex1
            else:
                vi = faces_copy[i_face].vertex3
                vj = faces_copy[i_face].vertex1
                vk = faces_copy[i_face].vertex2
            theta_ij = math.atan2(vj.y-vi.y, vj.x-vi.x )
            ri = vi.radius
            rj = vj.radius
            rk = vk.radius
            alpha_i = math.acos( ((ri+rj)**2+(ri+rk)**2-(rj+rk)**2)/(2*(ri+rj)* \
            (ri+rk)) )
            vk.x = vi.x + (ri+rk)*math.cos(alpha_i+theta_ij)
            vk.y = vi.y + (ri+rk)*math.sin(alpha_i+theta_ij)
            vk.placed = 1
            faces_copy.pop(i_face)
            i_face = 0
        else:
            i_face= i_face + 1
            if i_face == len(faces_copy):
                i_face = 0
    #adjust radii
    adjust_ratio = x_max/LD[0].x
    for c in circles:
        c.x = c.x * adjust_ratio
        c.y = c.y * adjust_ratio
        c.radius = c.radius * adjust_ratio
        c.placed = 0
    #draw circles
    #fig, ax = plt.subplots()
    #plt.figure(figsize=(15,15))
    fig = plt.gcf()
    ax = fig.gca()
    for c in circles:
        ax.add_artist( plt.Circle((c.x/l, c.y/l), c.radius/l, color='g') )
    plt.show()
    return circles, faces, LD, AD, cb, anchor_x, anchor_y, cb_origin


def generate_boundary_edges():
    circles,faces,LD,AD,Cb,anchor_x,anchor_y,cb_origin = generate_circlepacking()
    n_tri = len(faces)
    n_c = len(circles)
    # record old radii
    r_old = []
    for c in circles:
        r_old.append(c.radius)
    #randomly change radii of LD
    for c in LD:
        c.radius = c.radius*(1 -0.1 + 0.2*random.uniform(0,1))
    #Calculate radii for interior and undertermined circles
    eps = 0.001
    delta_r = 0.0001
    theta_diff = [0]*len(AD)
    for i in range(0, len(AD)):
        theta_diff[i] = theta_arround(AD[i]) - AD[i].totall_angle
    while np.max(theta_diff) > eps or np.min(theta_diff) < -eps:
        for i in range(0, len(AD)):
            if theta_diff[i] < 0:
                AD[i].radius = AD[i].radius - delta_r*AD[i].radius
            elif theta_diff[i] > 0:
                AD[i].radius = AD[i].radius + delta_r*AD[i].radius
        for i in range(0, len(AD)):
            theta_diff[i] = theta_arround(AD[i]) - AD[i].totall_angle
    #Layout circles
    faces_copy = faces.copy()
    #layout anchers along x
    x_coord = 0
    anchor_x[0].x = 0
    anchor_x[0].y = 0
    anchor_x[0].placed = 1      
    for i in range(1, len(anchor_x)):
        x_coord = x_coord + anchor_x[i-1].radius + anchor_x[i].radius
        anchor_x[i].x = x_coord
        anchor_x[i].y = 0
        anchor_x[i].placed = 1
    #layout anchers along y
    y_coord = 0
    y_coord = y_coord + cb_origin.radius + anchor_y[0].radius
    anchor_y[0].y = y_coord;
    anchor_y[0].x = 0;
    anchor_y[0].placed = 1;
    for i in range(1, len(anchor_y)):
        y_coord = y_coord + anchor_y[i-1].radius + anchor_y[i].radius;
        anchor_y[i].y = y_coord;
        anchor_y[i].x = 0;
        anchor_y[i].placed = 1;
    #layour other circles
    i_face=0
    while faces_copy != []:
        if faces_copy[i_face].vertex1.placed + faces_copy[i_face].vertex2.placed + \
        faces_copy[i_face].vertex3.placed == 3:
            faces_copy.pop(i_face)
            i_face = 0
        elif faces_copy[i_face].vertex1.placed + faces_copy[i_face].vertex2.placed + \
        faces_copy[i_face].vertex3.placed == 2:
            if faces_copy[i_face].vertex1.placed == 1 and \
            faces_copy[i_face].vertex2.placed == 1:
                vi = faces_copy[i_face].vertex1
                vj = faces_copy[i_face].vertex2
                vk = faces_copy[i_face].vertex3
            elif faces_copy[i_face].vertex2.placed == 1 and \
            faces_copy[i_face].vertex3.placed == 1:
                vi = faces_copy[i_face].vertex2
                vj = faces_copy[i_face].vertex3
                vk = faces_copy[i_face].vertex1
            else:
                vi = faces_copy[i_face].vertex3
                vj = faces_copy[i_face].vertex1
                vk = faces_copy[i_face].vertex2
            theta_ij = math.atan2(vj.y-vi.y, vj.x-vi.x )
            ri = vi.radius
            rj = vj.radius
            rk = vk.radius
            alpha_i = math.acos( ((ri+rj)**2+(ri+rk)**2-(rj+rk)**2)/(2*(ri+rj)* \
            (ri+rk)) )
            vk.x = vi.x + (ri+rk)*math.cos(alpha_i+theta_ij)
            vk.y = vi.y + (ri+rk)*math.sin(alpha_i+theta_ij)
            vk.placed = 1
            faces_copy.pop(i_face)
            i_face = 0
        else:
            i_face= i_face + 1
            if i_face == len(faces_copy):
                i_face = 0
    # plot circles
    fig = plt.gcf()
    ax = fig.gca()
    for c in circles:
        ax.add_artist( plt.Circle((c.x/20.0, c.y/20.0), c.radius/20.0, color='g') )
    plt.show()
    # output coordinate of LD
    boundary_edges = []
    for i in range(0, len(LD)-1):
        boundary_edges.append([1, LD[i].x, LD[i].y, LD[i+1].x, LD[i+1].y])
    # output radii of LD and all circles
    r_ld = []
    for c in LD:
        r_ld.append(c.radius)
    r = []
    for c in circles:
        r.append(c.radius)
    # unplace and resum circles
    for i in range(0, len(circles)):
        circles[i].placed = 0
        circles[i].radius = r_old[i]
    return boundary_edges, r_ld, r



        

    