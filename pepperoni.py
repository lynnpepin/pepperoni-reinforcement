# Pepperoni module
"""
Data structures: CircleVertex, HalfEdge, Face
Functions:
bool = in_circles(c, circles)
float = theta_arround(c)
LD, AD, circles, faces, cb, anchor_x, anchor_y, origin = generate_circlepacking()
modify_circlepacking(dr, LD, AD)
boundar_edges, r_ld, r = generate_boundary_edges(LD, AD, circles, faces, cb, anchor_x, anchor_y, origin)
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
    def __init__(self, i=[], r=[], ang=[], x=[], y=[]):
        self.index = i # the index of the circle in th list of circles
        self.radius = r
        self.totall_angle = ang # the surround angle of the circle. 2pi for
        self.x = x                            # interior circle
        self.y = y
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
    return tri  

def faces_halfedges(tri, circles):
    """
    triples in circle packing are considered as faces
    halfedge is a kind of direct edge, source -> target
    
    # Arguments:
        tri: Instance of Triangulation
        circles: list of CircleVertex
        
    # Returns:
        faces: list of Face
        halfedges: list of HalfEdge
    """
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
    return faces, halfedges

def boundary(halfedges):
    """
    collect the boundary circles in ccw way. cb[0] == cb[-1]
    
    # Arguments:
        halfedges: list of HalfEdge
        
    # Returns:
        cb: list of CircleVertex, the boundary circles
    """
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
    return cb

def neighbors_ci(ci):
    """
    find the neighbor vertices for interior circles in ccw. the start and end 
    of the neigbors overlap.
    
    # Arguments:
        ci: list of CircleVertex: interiro circles. 
            ci.neighbors[0] == ci.neighbors[-1f]
        
    # Returns:
        none
    """
    #find neighbor for every interior circles
    for i in range(0,len(ci)):
        traveller2 = ci[i].incident_halfedge
        ci[i].neighbors = [traveller2.target]
        traveller2 = traveller2.next.next.flip
        ci[i].neighbors.append(traveller2.target)
        while ci[i].neighbors[-1].index != ci[i].neighbors[0].index:
            traveller2 = traveller2.next.next.flip
            ci[i].neighbors.append(traveller2.target)
            
def neighbors_cb(cb):
    """
    find the neighbor vertices for boundary circles in ccw. t
    
    # Arguments:
        cb: list of CircleVertex: boundary circles. 
        
    # Returns:
        none
    """
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
                
def ld_start_end_origin(cb, points):
    """
    find the start, end of leading dancers in boundary circles collections.
    find the circle on origin of coordinate
    
    # Arguments:
        cb: list of CircleVertex, the boundary circles
        points: the coordinates of center of circles
        
    # Returns:
        ld_start: CircleVertex, the start circle of leading dancers
        ld_start_index: int, the index of ld_start in cb
        ld_end: CircleVertex, the end circle of leading dancers
        ld_end_index: int, the index of ld_end in cb
        origin: the circle on the origin of coordinate
        origin_index_cb: the index of origin in cb
    """
    x_max = np.max(points[:,0])
    y_max = np.max(points[:,1])
    for i in range(0,len(cb)):
        if cb[i].x == x_max and cb[i].y == 0:
            ld_start = cb[i]
            ld_start_index = i
        elif cb[i].x == 0 and cb[i].y == y_max:
            ld_end = cb[i]
            ld_end_index = i
        elif cb[i].x == 0 and cb[i].y == 0:
            origin = cb[i]
            origin_index_cb = i
    return ld_start, ld_start_index, ld_end, ld_end_index, origin, origin_index_cb

def leanding_dancers(cb, ld_start_index, ld_end_index):
    """
    collect the leading dancers (the circles we can manipulate their radii)
    
    # Arguments:
        cb: list of CircleVertex, the boundary circles
        ld_start_index: int, the index of ld_start in cb
        ld_end_index: int, the index of ld_end in cb 
    """
    j = ld_start_index
    LD = []
    while cb[j] != cb[ld_end_index]:
        LD.append(cb[j])
        j = (j+1) % (len(cb)-1)
    LD.append(cb[ld_end_index])
    return LD

def calculate_ld_surround_angles(LD):
    """
    the initial shape of bridge hole determined by the the surround angle of
    leading dancers
    
    # Arguments:
        LD: list of CircleVertex, leading dancers
        
    # Returns:
        none
    """
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
        
def calculate_radii(circles, eps, delta_r, leavingout=[]):
    """
    calculate the radii of circles by adjusting radii untill the surround_angles 
    of each circle are equal with the totall anges that prescribe at first
    If all the radii of boundary circle are undertmined, must leaving one boundary
    circle out.
    
    # Arguments:
        circles: list of CircleVertex
        eps: float, error tolerate 
        delta_r: float, the magnitude of radii change in each iteration
        leavingout: CircleVertex, one of the boundary circle
        
    # Returns:
        none
    """
    n_c = len(circles)
    theta_diff = [0]*n_c # the difference between the expected angle and actual angle 
    for i in range(0,n_c):
        if circles[i].index != leavingout.index:
            theta_diff[i] = theta_arround(circles[i]) - circles[i].totall_angle
    while np.max(theta_diff) > eps or np.min(theta_diff) < -eps:
        for i in range(0, n_c):
            if theta_diff[i] < 0:
                if circles[i].index != leavingout.index:
                    circles[i].radius = circles[i].radius - delta_r*circles[i].radius
            elif theta_diff[i] > 0:
                if circles[i].index != leavingout.index:
                    circles[i].radius = circles[i].radius + delta_r*circles[i].radius
        for i in range(0, n_c):
            if circles[i].index != leavingout.index:
                theta_diff[i] = theta_arround(circles[i]) - circles[i].totall_angle

def anchor_x_y(cb, origin_index_cb):
    """
    Fix the circles lie along x axis and y axis during laying out circles
    
    # Arguments:
        cb: list of CircleVertex, boundary circles
        origin_index_cb, the index of circle lying on origin in cb
        
    # Returns:
        anchor_x: list of CircleVertex, the circles lying along x axis 
                including origin
        anchor_y: listn of CircleVertex, the circle lying along y axis
    """
    i = origin_index_cb
    anchor_x = []
    while cb[i].y == 0:
        anchor_x.append(cb[i])
        i = i+1
        if i == len(cb):
            i = 1
    #anchers lie along y axis
    i = origin_index_cb-1
    if i < 0:
        i = len(cb)-2
    anchor_y = []
    while cb[i].x == 0:
        anchor_y.append(cb[i])
        i = i-1
        if i < 0:
            i = len(cb) - 2
    return anchor_x, anchor_y

def layout_circles(faces, anchor_x, anchor_y, origin):
    """
    layout circles
    
    # Arguments:
        faces: list of Faces
        anchor_x: list of CircleVertex, the circles lying along x axis
        anchor_y: list of CircleVertex, the circles lying along y axis
        origin: CircleVertex, the cirlce lying on origin of coordinate
    """
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
    y_coord = y_coord + origin.radius + anchor_y[0].radius
    anchor_y[0].y = y_coord;
    anchor_y[0].x = 0;
    anchor_y[0].placed = 1;
    for i in range(1, len(anchor_y)):
        y_coord = y_coord + anchor_y[i-1].radius + anchor_y[i].radius;
        anchor_y[i].y = y_coord;
        anchor_y[i].x = 0;
        anchor_y[i].placed = 1;
    #layour other circles
    faces_copy = faces.copy()
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
                
def draw_circles(circles,l):
    """
    draw circles
    
    # Arguments:
        circles: list of CircleVertex
        l: float, the half length of bridge
    """
    fig = plt.gcf()
    ax = fig.gca()
    for c in circles:
        ax.add_artist( plt.Circle((c.x/l, c.y/l), c.radius/l, color='g') )
    plt.show()
              
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
    return tri   

def generate_circlepacking(tri, delta, eps, delta_r):
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
    LD: List of CircleVertex, list of leading dancers
    AD: List of CircleVertex, list of accompanying dancers
    circles: List of CircleVertex, list of all circles
    faces: List of Face
    cb: List of CircleVertex, list of boundary circles
    anchor_x: List of CircleVertex, list of the circles lying along x axis
    anchor_y: List of CircleVertex, list of the circles lying along y axis
    origin: CircleVertex, the circle lying on the origin (0,0)
    """
    circles = [] # the circles of the packing
    n_c = len(tri.points)
    for i in range(0, n_c): # initialization of the circles
        x = tri.points[i][0]
        y = tri.points[i][1]
        if x == 0 and y > 0:
            circles.append(CircleVertex(i, delta/2, np.pi, x, y))
        elif x > 0 and y == 0:
            circles.append(CircleVertex(i, delta/2, np.pi, x, y))
        elif x == 0 and y == 0:
            circles.append(CircleVertex(i, delta/2, np.pi/2, x, y))
        else:
            circles.append(CircleVertex(i, delta/2,0, x, y))
    faces, halfedges = faces_halfedges(tri, circles)# the faces and halfedges of the circle packing
    cb = boundary(halfedges) # collection of boundary circles
    ci = [] # collection of interior circles
    for c in circles:
        if in_circles(c, cb) == False:
            c.totall_angle = 2*np.pi
            ci.append(c)
    #find neighbor for every interior circles
    neighbors_ci(ci)
    #find neighbor for every boundary circles
    neighbors_cb(cb)
    #find the start point and the end point of the sequence of leading dancers
    ld_start, ld_start_index, ld_end, ld_end_index, origin, origin_index_cb = \
    ld_start_end_origin(cb, tri.points)
    #collect the leading dancers (the circles we can manipulate their radii)
    LD = leanding_dancers(cb, ld_start_index, ld_end_index)
    #collect the accompanying dancers (the rest of the circles)
    AD = []
    for c in circles:
        if in_circles(c, LD) == False:
            AD.append(c)
    #determine the surround angle for leading dancers
    calculate_ld_surround_angles(LD)
    #Calculate radii for circles
    calculate_radii(circles, eps, delta_r, LD[1])
    #Layout circles
    #anchers lie along x axis
    anchor_x, anchor_y = anchor_x_y(cb, origin_index_cb)
    #layout anchers along x
    layout_circles(faces, anchor_x, anchor_y, origin)
     # unplace circles
    for i in range(0, len(circles)):
        circles[i].placed = 0
    #adjust radii and unplace circles
    x_max = max(tri.points[:,0])
    adjust_ratio = x_max/LD[0].x
    for c in circles:
        c.x = c.x * adjust_ratio
        c.y = c.y * adjust_ratio
        c.radius = c.radius * adjust_ratio
    return LD, AD, circles, faces, cb, anchor_x, anchor_y, origin


def modify_circlepacking(dr,LD, AD, eps, delta_r):
    """
    Add modification to the radii of leading dancers, then calculate
    the radii of accompanying dancers
    
    # Arguments:
        dr: list of float, the changes added to radii of leading dancers
        LD: list of CircleVertex, the list of leading dancers
        AD: list of CircleVertex, the list of accompanying dancers
    
    # Returns:
    no reuturn
    """
    for i in range(0, len(LD)): # add the changes
        LD[i].radius = LD[i].radius + dr[i]
    calculate_radii(AD, eps, delta_r)
            
def generate_boundary_edges(LD, circles, faces, anchor_x, anchor_y, origin):
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
        origin: CircleVertex, the circle lying on the origin (0,0)
        
    # Returns:
        boundary_edges: nx5 Arrary, n is the length of LD
        r_ld: list of float, the collection of radii of leading dancers
        r: list of float, the collection of all circles
    """
    #Layout circles
    layout_circles(faces, anchor_x, anchor_y, origin)
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





        

    