"""Circle-packing related utilities."""

import math
import random
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from _FEM import _ccw, _membershiptest, _FEM


class BridgeHoleDesign:
    l = 20  # the half length of the bridge
    h = 10  # the height of the bridge
    a_ell = 16.0  # the initial shape of the hole is an ellipse
    b_ell = 8.0  # x^2/a_ell^2 + y^2/b_ell^2 = 1
    delta = 1.0  # distance between the points of triangulation
    eps = 0.01  # error tolerate for circle packing calculation
    delta_r = 0.001  # the changes in each iteration in the process of calculation
    nely = 10  # the number of elements in y direction for FEM
    nelx = 20  # the number of elements in x direction for FEM
    ri = []  # List of float > 0, the radii of interior circles
    raccb = []  # List of float > 0, the radii of accompanying boundary circles
    rld = []  # List of float > 0, the radii of leading dancers
    r = []  # List of float > 0, the radii of all circles
    sigma = []  # float, the maximum stress in the bridge under the loads
    area = []  # float, the area of the hole
    mass = []  # float, the mass of the bridge, density is 1
    gmass_r = []  # list of float, the gradient of mass resprect to all radii
    gmass_rld = []  # list of float, the gradient of mass repsect ot leading dancers
    angles_ld = []  # list of float, the surround angles of leading dancers
    angles_accb = []  # list of float, the surround angles of accompanying boundary dancers
    angles_cb = []  # list of float, the surround angles of boundary circles
    total_length_ld = []  # float, the total edge length of leading dancers
    total_length_accb = []  # float, the total edge length of accompanying boundary dancers
    total_length_cb = []  # float, the total edge length of boundary circles
    edges_ld = []  # list of float, the edge list of leading dancers
    edges_accb = []  # list of float, the edge list of accompanying boundary dancers
    edges_cb = []  # list of float, the edge list of boundary circles
    positions_ld = []  # nX2 array, the x and y coordinates of center of leading dancers
    positions_accb = []  # nX2 array, the x and y coordinates of center of accompanying boundary dancers
    positions_ci = []  # nX2 array, the x and y coordinates of center of interior circles
    positions_cb = []  # nX2 array, the x and y coordinates of center of boundary circles
    positions_all = []  # nX2 array, the x and y coordinates of center of all circles
    _tri = []  # Delaunay, a triangluation instance
    _circles = []  # list of _CircleVertex, recording information of all circles
    _halfedges = []  # list of _HalfEdge, the halfedges in the triangulation
    _faces = []  # list of _Face, the faces in the triangulation
    _LD = []  # list of CircleVertex, collection of leading dancers
    _AccB = []  # list of CircleVertex, collection of accompanying boundary dancers
    _ci = []  # list of CircleVertex, collection of interior circles
    _AD = []  # list of CircleVertex, collection of accompanying dancers
    _cb = []  # list of CircleVertex, collection of boundary circles
    _anchor_x = []  # list of CircleVertex, the circles lying along x axis
    _anchor_y = []  # list of CircleVertex, the circles lying along y axis
    _cb_origin = []  # CircleVertex, the circle lying on origin of coordinate
    _edges = []  # nX5 arrary, the coordinates of leading dancers

    def __init__(self):
        """
        Initialize the circle packing based on preset triangulation.  
        
        """
        # generate the triangulation of the bridge hole in ellipitical shape
        # with a point gap as delta
        self._tri = _generate_triangulation(self.l, self.h, self.a_ell,
                                            self.b_ell, self.delta)
        # create the circle packing based on the triangulation created above
        self.rld, self.raccb, self.ri, self.r, self._LD, self._AccB, self._ci, self._AD, self._circles, self._faces, self._cb, self._anchor_x, self._anchor_y, self._cb_origin = _generate_circlepacking(
            self._tri, self.delta, self.eps, self.delta_r)
        # generate the boundary edges of the hole, which will be used in FEM
        self._edges = _generate_boundary_edges(self._LD, self._circles,
                                               self._faces, self._anchor_x,
                                               self._anchor_y, self._cb_origin)
        # Using FEM, calculating the maximum stress and the area of the hole
        self.sigma, self.area = _finite_element_analysis(
            self._edges, self.nely, self.nelx, self.l, self.h)
        # assuming the density is 1, calculating the mass of the bridge by deducting
        # the area of hole from the area of the rectangle
        self.mass = self.l * self.h - _get_area_of_all(self._faces)
        # calculate the gradient of the mass respect to r, and rld
        self.gmass_r, self.gmass_rld = _get_grad_mass(self.r, self._LD,
                                                      self._circles)
        # get the surround angles of leading dancers
        self.angles_ld = _get_surround_angles(self._LD)
        # get the surround angles of boundary circles
        self.angles_accb = _get_surround_angles(self._AccB)
        # get the length of edges linking by leading dancers
        self.total_length_ld, self.edges_ld = _get_edge_length(self._LD, 0)
        # get the length of edges linking by accompanying boundary dancers
        self.total_length_accb, self.edges_accb = _get_edge_length(
            self._AccB, 0)
        # get the positions of leading dancers
        self.positions_ld = _get_positions(self._LD)
        # get the positions of boundary circles
        self.positions_accb = _get_positions(self._AccB)
        # get the positions of interior circles
        self.positions_ci = _get_positions(self._ci)

    def update(self, rld_new):
        """
        To update circle packing with a new radii list for leading dancers
        # Arguments:
            rld_new: list of float > 0, a new radii list of leading dancers. Note that rld_new is recommened to be larger than 
            the precision self.eps.
            
        # Returns:
            self.r: list of float, the updated radii list of all circles
            self.sigma: float, the maximum stress of the updated design
            self.mass: float, the mass of the updated desigin
            self.gmass_r: list of float, the gradient of the mass respect to r
            self.gmass_rld: list of float, the gradient of the mass respect to rld
            self.angles_ld: list of float, 
        """
        # modify the cricle packing given a new radii of leading dancers
        _modify_circlepacking(rld_new, self.raccb, self.ri, self.r, self._LD,
                              self._AccB, self._ci, self._AD, self._circles,
                              0.1 * self.eps, 0.1 * self.delta_r)
        # modify the boudanry edge as the circle packing changes
        _modify_boundary_edges(self._edges, self._LD, self._circles,
                               self._faces, self._anchor_x, self._anchor_y,
                               self._cb_origin)

        self.sigma, self.area = _finite_element_analysis(
            self._edges, self.nely, self.nelx, self.l, self.h)

        self.mass = self.l * self.h - _get_area_of_all(self._faces)

        self.gmass_r, self.gmass_rld = _get_grad_mass(self.r, self._LD,
                                                      self._circles)

        self.angles_ld = _get_surround_angles(self._LD)

        self.angles_accb = _get_surround_angles(self._AccB)

        self.total_length_ld, self.edges_ld = _get_edge_length(self._LD, 0)

        self.total_length_accb, self.edges_accb = _get_edge_length(
            self._AccB, 0)
        # get the positions of leading dancers
        self.positions_ld = _get_positions(self._LD)
        # get the positions of boundary circles
        self.positions_accb = _get_positions(self._AccB)
        # get the positions of all circles
        self.positions_ci = _get_positions(self._ci)

        geo = {
            'angles_ld': self.angles_ld,
            'angles_accb': self.angles_accb,
            'total_length_ld': self.total_length_ld,
            'total_length_accb': self.total_length_accb,
            'edges_ld': self.edges_ld,
            'edges_accb': self.edges_accb,
            'positions_ld': self.positions_ld,
            'positions_accb': self.positions_accb,
            'positions_ci': self.positions_ci
        }

        data = {
            'raccb': self.raccb,
            'ri': self.ri,
            'sigma': self.sigma,
            'mass': self.mass,
            'gmass_r': self.gmass_r,
            'gmass_rld': self.gmass_rld,
            'geometry_info': geo
        }
        return data

    def draw_circlepacking(self):
        """
        draw the circle packing
        """
        _draw_circles(self._circles, self.l)

    def draw_triangulation(self):
        """
        draw the triangulation
        """
        _draw_triangulation(self._tri)


def _finite_element_analysis(edges, nely, nelx, l, h):
    """
    If there are parts of edges exceeding the design domain, return infinity 
    for stress, else calls _FEM 
    
    # Arguments:
    edges: nx5 float arrary. The line segments of the part of hole under designing
    nely: int, number of elements in y direction
    nelx: int, number of elements in x direction
    l: float, the length of the rectangle design domain
    h: float, the height of the rectangle design domain
    
    # Returns:
    sigma: float, the maximal stress of the bridge under loading
    area: float, the area of the bridge
    """
    eps_x = l / nelx
    eps_y = h / nely
    
    if edges[-1][3] > l - 1 * eps_x or \
       edges[-1][4] > h - 1 * eps_y:
        sigma = 10**10
        area = l * h
        return sigma, area
    
    for e in edges:
        if e[1] > l - 1 * eps_x or \
           e[2] > h - 1 * eps_y:
            sigma = 10**10
            area = l * h
            return sigma, area 
    
    sigma, area = _FEM(edges, nely, nelx, False)
    return sigma, area


def _get_area_ri(ri, rj, rk):
    """
    calculate the partial difference of the area of triangle determined 
    by the triple (ri, rj, rk) repsect to ri
    
    # Arguments:
        ri: float, the radius 
        rj: float, the radius
        rk: float, the radius
        
    # Return:
        the partial defference of the area respect to ri
    """
    return rj * rk * (2 * ri + rj + rk) / (2 * np.sqrt(ri * rj * rk *
                                                       (ri + rj + rk)))


def _get_grad_mass(r, LD, circles):
    """
    calculate the gradient of mass respect to radii of circles
    
    # Arguments
        r: list of float, the radii list of all circles
        LD: list of _CircleVertex, the collectin of leading dancers
        circles: list of _CircleVertex, the collection of all circles
        
    # Returns:
        gmass_r: list of float, the gradient of mass respect the radii of all 
                 circles
        gmass_rld: list of float, the gradient of mass respect to radii of leading
                    dancers        
    """
    gmass_r = [0] * len(r)
    for i_r in range(0, len(r)):
        # the partial derivative of hole area respect ri is the sum of the partial derivative
        # of the area of triangles that include ri respect ri. These trianges are the trianges
        # surrounding circles[i], which can be found be the neighbors of circles[i]
        for i_n in range(0, len(circles[i_r].neighbors) - 1):
            ri = r[i_r]
            rj = circles[i_r].neighbors[i_n].radius
            rk = circles[i_r].neighbors[i_n + 1].radius
            gmass_r[i_r] = gmass_r[i_r] - _get_area_ri(ri, rj, rk)
    gmass_rld = [0] * len(LD)
    for i in range(0, len(LD)):
        gmass_rld[i] = gmass_r[LD[i].index]
    return gmass_r, gmass_rld


def _get_area_of_all(faces):
    """
    calculate the area of the hole by summing up the are of all triangles
    
    # Arguments
        faces: list of _Face, the faces of the triangulation
        
    # Returns:
        s: float, the area
    """
    s = 0
    for f in faces:
        ri = f.vertex1.radius
        rj = f.vertex2.radius
        rk = f.vertex3.radius
        a = ri + rj
        b = rj + rk
        c = rk + ri
        p = (a + b + c) / 2
        s = s + np.sqrt(p * (p - a) * (p - b) * (p - c))
    return s


def _get_surround_angles(Cir):
    """
    Get the surround angles of a set of circles
    
    # Arguments:
    Cir: _CircleVertex array, the circle set
    
    # Returns:
    ang: float array, the array of the surround angle of the given circles
    """
    ang = [0] * len(Cir)
    for i in range(0, len(Cir)):
        ang[i] = _theta_arround(Cir[i])
    return ang


def _get_edge_length(Cir, closed):
    """
    Get the total length of the part of boundary linking by a set of circles and the lengths of the
    segments on the part boudnary
    
    # Arguments: 
    Cir: _CircleVertex array, the set of circles on the boundary
    closed: Bool, denote whether the set of circles are linked into a closed curve
    
    # Returns:
    total: float, the total length of the part of boundary
    edge_list: float array, the lengths of the segments linked between the circles 
    """
    edge_list = [0] * (len(Cir) - 1)
    total = 0
    for i in range(0, len(Cir) - 1):
        edge_list[i] = Cir[i].radius + Cir[i + 1].radius
        total = total + Cir[i].radius + Cir[i + 1].radius
    if closed == 1:
        edge_list.append(Cir[0].radius + Cir[-1].radius)
        total = total + Cir[0].radius + Cir[-1].radius
    return total, edge_list


def _get_positions(Cir):
    """
    Get the positions of a set of circles
    
    # Arguments:
    Cir: _CircleVertex array, the circle set
    
    # Returns:
    p: float array, nx[x,y], n is the number of circles. x and y are the coordinates
    of the centers of the circles
    """
    p = np.zeros((len(Cir), 2))
    for i in range(0, len(Cir)):
        p[i][0] = Cir[i].x
        p[i][1] = Cir[i].y
    return p


class _CircleVertex:
    """ The data structure for the circles in circle packing.

    # Properties
        index: int, the index of the circle in the circle packing
        radius: float > 0, the radius of the circle
        totall_angle: float, the surround angle of the circle
        x: float, the x coordinate of the circle
        y: float, the y coordinate of the circle

    # Example
    c is an instance of _CircleVertex
    c.index is the index of c in the circle packing. If c.index == 5, c is the fifthe
    circle in the circle packing.
    """
    # one of the halfedges incident with the circle
    incident_halfedge = []
    # the set of all of neighbors of the circle
    neighbors = []
    # whether the circle is placed or not
    placed = 0

    def __init__(self, i=[], r=[], ang=[], x=[], y=[]):
        # the index of the circle in th list of circles
        self.index = i
        self.radius = r
        # the surround angle of the circle. 2pi forinterior rcicle.
        self.totall_angle = ang
        self.x = x
        self.y = y


class _HalfEdge:
    """
    The data structure of the halfedge linked by two adjacent circles
    # Properties
        source: _CircleVertex
        target: _CircleVertex
        face_index: int, the index of face where the halfedge residents
        flip: _HalfEdge, the corresponding halfedge when flip the current halfedge
        next: _HalfEdge, the next halfedge of the current halfedge
        prev: _HalfEdge, the previous halfedge of the current halfedge

    # Example
    e = _HalfEdge(c1,c2,1)
    e.source = c1
    e.target = c2
    e.flip = _HalfEdge(c2,c1,2)
    """
    flip = []  # the corresponding halfedge when flip self
    next = []  # the next halfedge
    prev = []

    def __init__(self, vi=[], vj=[], f=[]):
        self.source = vi
        self.target = vj
        self.face_index = f


class _Face:
    """
    a face consits of three vertices and three halfedges that linking the vertices
    # Properties
       index: int, the index of the face in the faces list
       vertex1: _CircleVertex, the number 1 circle in the face
       vertex2: _CircleVertex, the number 2 circle in the face
       vertex3: _CircleVertex, the number 3 circle in the face
       halfedge1: _HalfEdge, the halfedge linking vertex1 and vertex2
       halfedge2: _HalfEdge, the halfedge linking vertex2 and vertex3
       halfedge3: _HalfEdge, the halfedge linking vertex3 and vertex1

    # Example
        f = _Face(c1,c2,c3,1)
    """

    def __init__(self, v1=[], v2=[], v3=[], i=[]):
        self.index = i  # index of the triangle in the triangle list
        self.vertex1 = v1
        self.vertex2 = v2
        self.vertex3 = v3
        self.halfedge1 = _HalfEdge(v1, v2, i)
        self.halfedge2 = _HalfEdge(v2, v3, i)
        self.halfedge3 = _HalfEdge(v3, v1, i)
        self.halfedge1.next = self.halfedge2
        self.halfedge2.next = self.halfedge3
        self.halfedge3.next = self.halfedge1
        if self.vertex1.incident_halfedge == []:
            self.vertex1.incident_halfedge = self.halfedge1
        if self.vertex2.incident_halfedge == []:
            self.vertex2.incident_halfedge = self.halfedge2
        if self.vertex3.incident_halfedge == []:
            self.vertex3.incident_halfedge = self.halfedge3


def _in_circles(c, Cir):
    """Determine whether _CircleVertex c in list of _CircleVertex Cir.

    # Arguments
        c: Instance of _CircleVertex
        Cir: List of _CircleVertex

    # Returns
        Bool

    # Example
        True == _in_circles(c1,[c1,c2,c3])
        False == _in_circles(c5,[c1,c2,c3])
    """
    for ci in Cir:
        if c.index == ci.index:
            return True

    return False


def _theta_arround(cv):
    """Calculate the surround angle of a circle vertex cv
    
    # Arguments
        cv: Instance of _CircleVertex

    # Returns
        theta, Float, the surround angle of cv. 

    # Example
        c = _CircleVertex(1,1,0,0,0)
        theta = _theta_arround(c)
        If c is an interior in a circle packing, theta should be 2*pi

    """
    r = cv.radius
    theta = 0
    for i in range(0, len(cv.neighbors) - 1):
        rj = cv.neighbors[i].radius
        rk = cv.neighbors[i + 1].radius
        # TODO - Better variable names here
        top = ((r + rj)**2 + (r + rk)**2 - (rj + rk)**2)
        bot = (2 * (r + rj) * (r + rk))
        val = top/bot
        theta = math.acos(top/bot) + theta
        # TODO - Remove
        # Example printings
        #except ValueError:
        #    print("####################")
        #    print(top, bot, val)
        #    print(r, rj, rk)
        #    print("####################")
        #    exit()
    return theta


def _generate_triangulation(l, h, a_ell, b_ell, delta):
    """The bridge has an elliptical hole
    discrete the elliptical hole with uniformed points in delta distance
    genetate the triangulation of these discrete points

    # Arguments
        l: float, the half length of the bridge, the deflaut value is 20
        h: float, the height of the bridge, the deflaut value is 10
        a_ell, b_ell: float the  parameters of ellipse, x^2/a_ell^2 + y^2/b_ell^2 = 1
            the deflaut value of a_ell is 18, the deflaut value of b_ell is 8
        delta: float, the distance of the discrete points in the domain, the deflaut value is 1

    # Returns
        tri: Instance of Triangulation
    """
    # x_domain, y_domain will be the set of x and y coordinates
    # of the points for triangulation
    x_domain = []
    y_domain = []
    x = np.linspace(0.0, l, math.ceil(l / delta))
    y = np.linspace(0.0, h, math.ceil(h / delta))
    for xi in x:
        for yi in y:
            if (xi / a_ell)**2 + (yi / b_ell)**2 < 1:
                x_domain.append(xi)
                y_domain.append(yi)
    # points = the points for trinagulation
    points = np.vstack([x_domain, y_domain]).T
    # generate the triangulation with scipy.Delaunay() and plot triangulation
    tri = Delaunay(points)  # generate the triangulation
    return tri


def _draw_triangulation(tri):
    plt.triplot(tri.points[:, 0], tri.points[:, 1], tri.simplices.copy())
    plt.plot(tri.points[:, 0], tri.points[:, 1], 'o')
    plt.show()


def _faces_halfedges(tri, circles):
    """
    Get the list of faces and the list of halfedges in the circle packing
    
    Triples in circle packing are considered as faces.
    _HalfEdge is a kind of direct edge, source -> target.

    # Arguments
        tri: Instance of Triangulation
        circles: list of _CircleVertex

    # Returns
        faces: list of _Face
        halfedges: list of _HalfEdge

    """
    # faces, halfedges are lists associated with the circle packing.
    faces = []
    halfedges = []

    n_t = len(tri.simplices)
    for i in range(0, n_t):  # create the faces with the circles
        faces.append(
            _Face(circles[tri.simplices[i][0]], circles[tri.simplices[i][1]],
                  circles[tri.simplices[i][2]], i))
        halfedges.append(faces[i].halfedge1)
        halfedges.append(faces[i].halfedge2)
        halfedges.append(faces[i].halfedge3)

    for j in range(len(halfedges)):  # collection of halfedges
        for i in range(len(halfedges)):
            if halfedges[i].source.index == halfedges[j].target.index and halfedges[i].target.index == halfedges[j].source.index:
                halfedges[j].flip = halfedges[i]

    return faces, halfedges


def _boundary(halfedges):
    """Collect the boundary circles in ccw way.
    The first circle and the last circle in the boundary circle list are same
    that is cb[0] == cb[-1]

    # Arguments
        halfedges: list of _HalfEdge, the list of halfedges in the circle packing

    # Returns
        cb: list of _CircleVertex, the boundary circles

    """
    cb = []  # collection of boundary circles
    for edge in halfedges: # find a halfedge that is on boundary
        if edge.flip == []:
            cb.append(edge.source)
            cb.append(edge.target)
            traveller = edge
            break
    while cb[-1].index != cb[0].index: 
        while traveller.next.flip != []:
            traveller = traveller.next.flip
        traveller = traveller.next
        cb.append(traveller.target)
    return cb


def _neighbors_ci(ci):
    """Find the neighbor vertices for interior circles in ccw.
    The start and end  of the neigbors overlap.
    That is, ci.neighbors[0] == ci.neighbors[-1f]

    TODO - Update explanation, comments within code.

    # Arguments
        ci: list of _CircleVertex: interiro circles. 

    # Example
    """
    # find neighbor for every interior circles
    for i in range(0, len(ci)):
        traveller2 = ci[i].incident_halfedge
        ci[i].neighbors = [traveller2.target]
        traveller2 = traveller2.next.next.flip
        ci[i].neighbors.append(traveller2.target)
        while ci[i].neighbors[-1].index != ci[i].neighbors[0].index:
            traveller2 = traveller2.next.next.flip
            ci[i].neighbors.append(traveller2.target)


def _neighbors_cb(cb):
    """Find the neighbor vertices for boundary circles in ccw.

    TODO - Update explanation, comments within code.

    # Arguments
        cb: list of _CircleVertex, boundary circles.

    # Example
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


def _ld_start_end_origin(cb, points):
    """
    Find the start and the end of leading dancers in boundary circles collections.
    And find the circle on origin of coordinate.

    TODO - Update explanation, comments within code.

    # Arguments
        cb: list of _CircleVertex, the boundary circles
        points: nx2 float array, the coordinates of center of circles

    # Returns
        ld_start: _CircleVertex, the start circle of leading dancers
        ld_start_index: int, the index of ld_start in cb
        ld_end: _CircleVertex, the end circle of leading dancers
        ld_end_index: int, the index of ld_end in cb
        origin: the circle on the origin of coordinate
        origin_index_cb: the index of origin in cb

    # Example
    """
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    for i in range(0, len(cb)):
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


def _leanding_dancers(cb, ld_start_index, ld_end_index):
    """
    Collect the leading dancers (the circles we can manipulate their radii) from the
    boudnary circles. Because leading dancers are a part of boundary circles

    TODO: Update explanation

    # Arguments
        cb: list of _CircleVertex, the boundary circles
        ld_start_index: int, the index of ld_start in cb
        ld_end_index: int, the index of ld_end in cb 

    # Returns
        Returns: 
        LD: list of _CircleVertex, the set of leading dancers

    # Example
    """
    j = ld_start_index
    LD = []
    while cb[j] != cb[ld_end_index]:
        LD.append(cb[j])
        j = (j + 1) % (len(cb) - 1)
    LD.append(cb[ld_end_index])
    return LD


def _calculate_ld_surround_angles(LD):
    """
    Calculate the surround angles for leading dancers.

    # Arguments
        LD: list of _CircleVertex, leading dancers

    # Example
    """
    for j in range(0, len(LD)):
        surround_angle = 0
        for i in range(0, len(LD[j].neighbors) - 1):
            L1 = np.sqrt((LD[j].x - LD[j].neighbors[i].x)**2 +
                         (LD[j].y - LD[j].neighbors[i].y)**2)
            L2 = np.sqrt((LD[j].x - LD[j].neighbors[i + 1].x)**2 +
                         (LD[j].y - LD[j].neighbors[i + 1].y)**2)
            L3 = np.sqrt((LD[j].neighbors[i].x - LD[j].neighbors[i + 1].x)**2 +
                         (LD[j].neighbors[i].y - LD[j].neighbors[i + 1].y)**2)
            alpha = math.acos((L1**2 + L2**2 - L3**2) / (2 * L1 * L2))
            surround_angle = surround_angle + alpha
        LD[j].totall_angle = surround_angle


def _calculate_radii(circles, eps, delta_r, leavingout=[]):
    """Calculate the radii of circles by adjusting radii untill the
    surround_angles of each circle are equal with the totall anges that
    prescribe at first
    If all the radii of boundary circle are undertmined, must leaving one boundary
    circle out.

    TODO - Update explanation, comments within code.

    # Arguments
        circles: list of _CircleVertex
        eps: float, error tolerate 
        delta_r: float, the magnitude of radii change in each iteration
        leavingout: _CircleVertex, one of the boundary circle

    # Example
    """
    n_c = len(circles)
    # theta_diff: The difference between the expected angle and actual angle
    theta_diff = [0] * n_c
    for i in range(0, n_c):
        if circles[i].index != leavingout.index:
            theta_diff[i] = _theta_arround(circles[i]) - circles[i].totall_angle
    while np.max(theta_diff) > eps or np.min(theta_diff) < -eps:

        for i in range(0, n_c):
            if theta_diff[i] < 0:
                if circles[i].index != leavingout.index:
                    circles[i].radius = circles[i].radius - delta_r * circles[i].radius
            elif theta_diff[i] > 0:
                if circles[i].index != leavingout.index:
                    circles[i].radius = circles[i].radius + delta_r * circles[i].radius
        for i in range(0, n_c):
            if circles[i].index != leavingout.index:
                theta_diff[i] = _theta_arround(
                    circles[i]) - circles[i].totall_angle


def _anchor_x_y(cb, origin_index_cb):
    """
    Fix the circles lie along x axis and y axis during laying out circles
    # Arguments
        cb: list of _CircleVertex, boundary circles
        origin_index_cb, the index of circle lying on origin in cb

    # Returns
        anchor_x: list of _CircleVertex, the circles lying along x axis 
                including origin
        anchor_y: listn of _CircleVertex, the circle lying along y axis
    """
    i = origin_index_cb
    anchor_x = []
    while cb[i].y == 0:
        anchor_x.append(cb[i])
        i = i + 1
        if i == len(cb):
            i = 1
    # anchers lie along y axis
    i = origin_index_cb - 1
    if i < 0:
        i = len(cb) - 2
    anchor_y = []
    while cb[i].x == 0:
        anchor_y.append(cb[i])
        i = i - 1
        if i < 0:
            i = len(cb) - 2
    return anchor_x, anchor_y


def _layout_circles(faces, anchor_x, anchor_y, origin):
    """
    Lay out circles. First place the circle on the origin of the cooridinate and the circles in anchor_x 
    and anchor_y. After these circles are placed, the rest circles would be placed forcedly by the connectivity
    relationship encoded in faces.
    # Arguments:
        faces: list of Faces
        anchor_x: list of _CircleVertex, the circles lying along x axis
        anchor_y: list of _CircleVertex, the circles lying along y axis
        origin: _CircleVertex, the cirlce lying on origin of coordinate
    """
    x_coord = 0
    anchor_x[0].x = 0
    anchor_x[0].y = 0
    anchor_x[0].placed = 1
    for i in range(1, len(anchor_x)):
        x_coord = x_coord + anchor_x[i - 1].radius + anchor_x[i].radius
        anchor_x[i].x = x_coord
        anchor_x[i].y = 0
        anchor_x[i].placed = 1
    # layout anchers along y
    y_coord = 0
    y_coord = y_coord + origin.radius + anchor_y[0].radius
    anchor_y[0].y = y_coord
    anchor_y[0].x = 0
    anchor_y[0].placed = 1
    for i in range(1, len(anchor_y)):
        y_coord = y_coord + anchor_y[i - 1].radius + anchor_y[i].radius
        anchor_y[i].y = y_coord
        anchor_y[i].x = 0
        anchor_y[i].placed = 1
    # layour other circles
    faces_copy = faces.copy()
    i_face = 0
    while faces_copy != []:
        if 3 == faces_copy[i_face].vertex1.placed + faces_copy[i_face].vertex2.placed + faces_copy[i_face].vertex3.placed:
            faces_copy.pop(i_face)
            i_face = 0
        elif faces_copy[i_face].vertex1.placed + faces_copy[i_face].vertex2.placed + faces_copy[i_face].vertex3.placed == 2:
            if faces_copy[i_face].vertex1.placed == 1 and faces_copy[i_face].vertex2.placed == 1:
                vi = faces_copy[i_face].vertex1
                vj = faces_copy[i_face].vertex2
                vk = faces_copy[i_face].vertex3
            elif faces_copy[i_face].vertex2.placed == 1 and faces_copy[i_face].vertex3.placed == 1:
                vi = faces_copy[i_face].vertex2
                vj = faces_copy[i_face].vertex3
                vk = faces_copy[i_face].vertex1
            else:
                vi = faces_copy[i_face].vertex3
                vj = faces_copy[i_face].vertex1
                vk = faces_copy[i_face].vertex2
            theta_ij = math.atan2(vj.y - vi.y, vj.x - vi.x)
            ri = vi.radius
            rj = vj.radius
            rk = vk.radius
            alpha_i = math.acos(((ri + rj)**2 + (ri + rk)**2 - (rj + rk)**2) /
                                (2 * (ri + rj) * (ri + rk)))
            vk.x = vi.x + (ri + rk) * math.cos(alpha_i + theta_ij)
            vk.y = vi.y + (ri + rk) * math.sin(alpha_i + theta_ij)
            vk.placed = 1
            faces_copy.pop(i_face)
            i_face = 0
        else:
            i_face = i_face + 1
            if i_face == len(faces_copy):
                i_face = 0


def _draw_circles(circles, l):
    """ 
    Plot circles with their position and the position of their centers
    # Arguments:
        circles: list of _CircleVertex
        l: float, the half length of bridge
    """
    fig, ax = plt.subplots()
    # change default range so that new circles will work
    ax.set_xlim((0, l))
    ax.set_ylim((0, l))

    for c in circles:
        ax.add_artist(plt.Circle((c.x, c.y), c.radius, color='g'))
    plt.show()


def _generate_circlepacking(tri, delta, eps, delta_r):
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
    LD: List of _CircleVertex, list of leading dancers
    AD: List of _CircleVertex, list of accompanying dancers
    circles: List of _CircleVertex, list of all circles
    faces: List of _Face
    cb: List of _CircleVertex, list of boundary circles
    anchor_x: List of _CircleVertex, list of the circles lying along x axis
    anchor_y: List of _CircleVertex, list of the circles lying along y axis
    origin: _CircleVertex, the circle lying on the origin (0,0)
    """
    # the circles of the packing
    circles = []
    n_c = len(tri.points)
    # initialization of the circles
    for i in range(0, n_c):
        x = tri.points[i][0]
        y = tri.points[i][1]
        if x == 0 and y > 0:
            circles.append(_CircleVertex(i, delta / 2, np.pi, x, y))
        elif x > 0 and y == 0:
            circles.append(_CircleVertex(i, delta / 2, np.pi, x, y))
        elif x == 0 and y == 0:
            circles.append(_CircleVertex(i, delta / 2, np.pi / 2, x, y))
        else:
            circles.append(_CircleVertex(i, delta / 2, 0, x, y))
    # the faces and halfedges of the circle packing
    faces, halfedges = _faces_halfedges(tri, circles)
    cb = _boundary(halfedges)  # collection of boundary circles
    ci = []  # collection of interior circles
    for c in circles:
        if _in_circles(c, cb) == False:
            c.totall_angle = 2 * np.pi
            ci.append(c)
    # find neighbor for every interior circles
    _neighbors_ci(ci)
    # find neighbor for every boundary circles
    _neighbors_cb(cb)
    # find the start point and the end point of the sequence of leading dancers
    ld_start, ld_start_index, ld_end, ld_end_index, origin, origin_index_cb = _ld_start_end_origin(
        cb, tri.points)
    # collect the leading dancers (the circles we can manipulate their radii)
    LD = _leanding_dancers(cb, ld_start_index, ld_end_index)
    # collect the accompanying dancers (the rest of the circles)
    AD = []
    for c in circles:
        if _in_circles(c, LD) == False:
            AD.append(c)
    # determine the surround angle for leading dancers
    _calculate_ld_surround_angles(LD)
    # Calculate radii for circles
    _calculate_radii(circles, eps, delta_r, LD[1])
    # Layout circles
    # anchers lie along x axis
    anchor_x, anchor_y = _anchor_x_y(cb, origin_index_cb)
    # layout anchers along x
    _layout_circles(faces, anchor_x, anchor_y, origin)
    # unplace circles
    for i in range(0, len(circles)):
        circles[i].placed = 0
    # adjust radii and unplace circles
    x_max = max(tri.points[:, 0])
    adjust_ratio = x_max / LD[0].x
    for c in circles:
        c.x = c.x * adjust_ratio
        c.y = c.y * adjust_ratio
        c.radius = c.radius * adjust_ratio
    # output radii of LD, boundary and all circles
    rld = [0] * len(LD)
    for i in range(0, len(LD)):
        rld[i] = LD[i].radius
    # collect accomany boundary dancers
    AccB = []
    for i in range(0, len(cb) - 1):  # cb[0] == cb[-1]
        if _in_circles(cb[i], LD) == False:
            AccB.append(cb[i])
    raccb = [0] * len(AccB)
    for i in range(0, len(AccB)):
        raccb[i] = AccB[i].radius
    ri = [0] * len(ci)
    for i in range(0, len(ci)):
        ri[i] = ci[i].radius
    r = [0] * len(circles)
    for i in range(0, len(circles)):
        r[i] = circles[i].radius

    return rld, raccb, ri, r, LD, AccB, ci, AD, circles, faces, cb, anchor_x, anchor_y, origin


def _modify_circlepacking(rld_new, raccb, ri, r, LD, AccB, ci, AD, circles, eps,
                          delta_r):
    """
    Add modification to the radii of leading dancers, then calculate
    the radii of accompanying dancers

    # Arguments:

        dr: list of float, the changes added to radii of leading dancers
        LD: list of _CircleVertex, the list of leading dancers
        AD: list of _CircleVertex, the list of accompanying dancers
    """
    for i in range(0, len(LD)):  # add the changes
        LD[i].radius = rld_new[i]
    _calculate_radii(AD, eps, delta_r)
    # output radii of all circles
    for i in range(0, len(AccB)):
        raccb[i] = AccB[i].radius
    for i in range(0, len(ci)):
        ri[i] = ci[i].radius
    for i in range(0, len(circles)):
        r[i] = circles[i].radius


def _generate_boundary_edges(LD, circles, faces, anchor_x, anchor_y, origin):
    """
    layout the circlepacking, determine the coordintes for every circle, then
    get the coordinates of the points on boundary edges

    # Arguments:
        LD: List of _CircleVertex, collection of leading dancers
        AD: List of _CircleVertex, collection of accompanying dancers
        circles: List of _CircleVertex, collection of all circles
        faces: List of _Face
        cb: List of _CircleVertex, collection of boundary circles
        anchor_x: List of _CircleVertex, collection of the circles lying along x axis
        anchor_y: List of _CircleVertex, collection of the circles lying along y axis
        origin: _CircleVertex, the circle lying on the origin (0,0)

    # Returns:
        boundary_edges: nx5 Arrary, n is the length of LD
        r_ld: list of float, the collection of radii of leading dancers
        r: list of float, the collection of all circles

    """
    # Layout circles
    _layout_circles(faces, anchor_x, anchor_y, origin)
    # unplace circles
    for i in range(0, len(circles)):
        circles[i].placed = 0
    # output coordinate of LD
    boundary_edges = np.ones((len(LD) - 1, 5))
    for i in range(0, len(LD) - 1):
        boundary_edges[i][1] = LD[i].x
        boundary_edges[i][2] = LD[i].y
        boundary_edges[i][3] = LD[i + 1].x
        boundary_edges[i][4] = LD[i + 1].y
    return boundary_edges


def _modify_boundary_edges(boundary_edges, LD, circles, faces, anchor_x,
                           anchor_y, origin):
    """
    layout the circlepacking, determine the coordintes for every circle, then
    get the coordinates of the points on boundary edges
    """
    # Layout circles
    _layout_circles(faces, anchor_x, anchor_y, origin)
    # unplace circles
    for i in range(0, len(circles)):
        circles[i].placed = 0
    # output coordinate of LD
    for i in range(0, len(boundary_edges) - 1):
        boundary_edges[i][1] = LD[i].x
        boundary_edges[i][2] = LD[i].y
        boundary_edges[i][3] = LD[i + 1].x
        boundary_edges[i][4] = LD[i + 1].y
