# Pepperoni module
"""
Data structures: CircleVertex, HalfEdge, Face
Functions:
bool = in_circles(c, circles)
float = theta_arround(c)
LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin = generate_circlepacking()
modify_circlepacking(dr, LD, AD)
boundar_edges, r_ld, r = generate_boundary_edges(LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin)
"""
import math
import random
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
class CircleVertex:
    """
    the data structure for the circles in circle packing
    """
    incident_halfedge = [] #one of the halfedges incident with the circle 
    neighbors = [] #the set of all of neighbors of the circle
    placed = 0 # whether the circle is placed or not
    def __init__(self, i=[], r=[], ang=[]):
        self.index = i # the index of the circle in th list of circles
        self.radius = r
        self.totall_angle = ang # the surround angle of the circle. 2pi for
                                # interior circle
class HalfEdge:
    flip = [] # the corresponding halfedge when flip self
    next = [] # the next halfedge 
    prev = []

    def __init__(self, vi=[], vj=[], f=[]):
        self.source = vi
        self.target = vj
        self.face_index = f

class Face:
    def __init__(self, v1=[], v2=[], v3=[], i=[]):
        self.index = i # index of the triangle in the triangle list
        self.vertex1 = v1
        self.vertex2 = v2
        self.vertex3 = v3
        self.halfedge1 = HalfEdge(v1, v2, i)
        self.halfedge2 = HalfEdge(v2, v3, i)
        self.halfedge3 = HalfEdge(v3, v1, i)
        self.halfedge1.next=self.halfedge2
        self.halfedge2.next=self.halfedge3
        self.halfedge3.next=self.halfedge1
        if self.vertex1.incident_halfedge == []:
            self.vertex1.incident_halfedge = self.halfedge1
        if self.vertex2.incident_halfedge == []:
            self.vertex2.incident_halfedge = self.halfedge2
        if self.vertex3.incident_halfedge == []:
            self.vertex3.incident_halfedge = self.halfedge3


def in_circles(c,Cir):
    """
    determine whether c in circle list
    
    # Arguments:
        c: Instance of CircleVertex
        Cir: List of CircleVertex
        
    # Returns:
        Bool if c can be found in Cir
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
    
    # Arguments:
        cv: Instance of CircleVertex
    
    # Returns:
        Float, the surround angle of cv. 
    CircleVertex -> float
    e.g. thera_arround(c) -> 3.14
    """
    r = cv.radius
    theta = 0
    for i in range(0,len(cv.neighbors)-1 ):
        rj = cv.neighbors[i].radius
        rk = cv.neighbors[i+1].radius
        theta = math.acos( ( (r+rj)**2 + (r+rk)**2 - 
        (rj+rk)**2 )/( 2*(r+rj)*(r+rk) ) ) + theta
    return theta

def generate_triangulation(l, h, a_ell, b_ell, delta):
    """
    the bridge has an elliptical hole
    discrete the elliptical hole with uniformed points in delta distance
    genetate the triangulation of these discrete points
    
    # Arguments:
        l: the half length of the bridge
        h: the height of the bridge
        a_ell, b_ell, the  parameters of ellipse, x^2/a_ell^2 + y^2/b_ell^2 = 1
        delta: the distance of the discrete points in the domain
    
    # Returns:
        tri: Instance of Triangulation
        x_domain: List of float, the x coordinates of the discrete points
        y_domain: List of float, the y coordinates of the discrete points
        
        
    """
    x_domain = [] # the set of x coordinate of the points for triangulation
    y_domain = [] # the set of y coordinate of the points for triangulation
    x = np.linspace(0.0, l, math.ceil(l/delta))
    y = np.linspace(0.0, h, math.ceil(h/delta))
    for xi in x:
        for yi in y:
            if (xi/a_ell)**2+(yi/b_ell)**2 < 1:
                x_domain.append(xi)
                y_domain.append(yi)
    points = np.vstack([x_domain,y_domain]).T # the points for triangulation
    tri = Delaunay(points) # generate the triangulation
    plt.triplot(points[:,0], points[:,1], tri.simplices.copy()) # plot the triangulation
    plt.plot(points[:,0], points[:,1], 'o')
    plt.show()
    return tri, x_domain, y_domain    

def generate_circlepacking(tri, x_domain, y_domain, delta, eps, delta_r):
    """
    generate a circlepacking whose complex K is same as the connectivity relation
    of tri
    
    # Arguments:
        tri: Instance of Triangulation
        x_domain: List of float, the x coordinates of the discrete points
        y_domain: List of float, the y coordinates of the discrete points
        delta: the distance of the discrete points in the domain
        eps: Float, error tolerate for circle packing calculation
        delta_r, Float, the changes of radii in each iteration in the process of calculation
        
    # Returns:
    LD: List of CircleVertex, collection of leading dancers
    AD: List of CircleVertex, collection of accompanying dancers
    circles: List of CircleVertex, collection of all circles
    faces: List of Face
    cb: List of CircleVertex, collection of boundary circles
    anchor_x: List of CircleVertex, collection of the circles lying along x axis
    anchor_y: List of CircleVertex, collection of the circles lying along y axis
    cb_origin: CircleVertex, the circle lying on the origin (0,0)
    """
    circles = [] # the circles of the packing
    n_c = len(x_domain)
    for i in range(0, n_c):
        circles.append(CircleVertex())
    for i in range(0, n_c): # initialization of the circles
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
    faces = [] # the faces of the circle packing
    halfedges = [] # the halfedges of the circle packing
    n_t = len(tri.simplices)
    for i in range(0, n_t): # create the faces with the circles
        faces.append(Face(circles[tri.simplices[i][0]], circles[tri.simplices[i][1]], 
        circles[tri.simplices[i][2]], i))
        halfedges.append(faces[i].halfedge1)
        halfedges.append(faces[i].halfedge2)
        halfedges.append(faces[i].halfedge3)
    for j in range(0, len(halfedges)): # collection of halfedges
        for i in range(0, len(halfedges)):
            if halfedges[i].source.index == halfedges[j].target.index and \
               halfedges[i].target.index == halfedges[j].source.index:
               halfedges[j].flip=halfedges[i]
    cb = [] # collection of boundary circles
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
    ci = [] # collection of interior circles
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
    #find the start point and the end point of the sequence of leading dancers
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
    #collect the leading dancers (the circles we can manipulate their radii)
    j = cb_start_index
    LD = []
    while cb[j] != cb_end:
        LD.append(cb[j])
        j = (j+1) % (len(cb)-1)
 #       if j == len(cb):
  #          j = 1
    LD.append(cb_end)
    #collect the accompanying dancers (the rest of the circles)
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
    #Calculate radii for circles
    theta_diff = [0]*n_c # the difference between the expected angle and actual angle 
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
    fig = plt.gcf()
    ax = fig.gca()
    for c in circles:
        ax.add_artist( plt.Circle((c.x/20, c.y/20), c.radius/20, color='g') )
    plt.show()
    return LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin


def modify_circlepacking(dr,LD, AD):
    """
    Add modification to the radii of leading dancers, then calculate
    the radii of accompanying dancers
    
    # Arguments:
        dr: list of float, the changes added to radii of leading dancers
        LD: list of CircleVertex, the collection of leading dancers
        AD: list of CircleVertex, the collection of accompanying dancers
    
    # Returns:
    no reuturn
    """
    for i in range(0, len(LD)): # add the changes
        LD[i].radius = LD[i].radius + dr[i]
    eps = 0.001 # the error tolerate
    delta_r = 0.0001 # the changes of radii in every iteration
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
            
            
def generate_boundary_edges(LD, AD, circles, faces, cb, anchor_x, anchor_y, cb_origin):
    """
    layout the circlepacking, determine the coordintes for every circle, then
    get the coordinates of the points on boundary edges
    
    # Arguments:
        LD: List of CircleVertex, collection of leading dancers
        AD: List of CircleVertex, collection of accompanying dancers
        circles: List of CircleVertex, collection of all circles
        faces: List of Face
        cb: List of CircleVertex, collection of boundary circles
        anchor_x: List of CircleVertex, collection of the circles lying along x axis
        anchor_y: List of CircleVertex, collection of the circles lying along y axis
        cb_origin: CircleVertex, the circle lying on the origin (0,0)
        
    # Returns:
        boundary_edges: nx5 Arrary, n is the length of LD
        r_ld: list of float, the collection of radii of leading dancers
        r: list of float, the collection of all circles
    """
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
            faces_copy.pop(i_face) # if the three circles in a face are all placed
            i_face = 0             # remove the face from faces_copy
        elif faces_copy[i_face].vertex1.placed + faces_copy[i_face].vertex2.placed + \
        faces_copy[i_face].vertex3.placed == 2: # if two circles in a face are placed place the rest one
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
        else: # in other cases, go to next face
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
    boundary_edges = np.ones((len(LD)-1,5))
    for i in range(0, len(LD)-1):
        boundary_edges[i][1] =  LD[i].x
        boundary_edges[i][2] =  LD[i].y
        boundary_edges[i][3] =  LD[i+1].x
        boundary_edges[i][4] =  LD[i+1].y
    # output radii of LD and all circles
    r_ld = []
    for c in LD:
        r_ld.append(c.radius)
    r = []
    for c in circles:
        r.append(c.radius)
    # unplace circles
    for i in range(0, len(circles)):
        circles[i].placed = 0
    return boundary_edges, r_ld, r



        

    