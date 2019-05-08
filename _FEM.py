# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:41:24 2019

@author: mmb18008
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from time import time
from scipy.sparse.linalg import cg
from config_dict import CONFIG

def _old_ccw(ax, ay, bx, by, cx, cy):
    return np.linalg.det([[ax, bx, cx],
                          [ay, by, cy],
                          [1,   1,  1]]) > 0

def _ccw(ax, ay, bx, by, cx, cy):
    """
    Determine wheter three points, a, b, and c are ordered in counter-clockwise(ccw) way
    
    # Arguments:
    ax: float, the x coordinate of point a
    ay: float, the y coordinate of point a
    bx: float, the x coordinate of point b
    by: float, the y coordinate of point b
    cx: float, the x coordinate of point c
    cy: float, the y coordinate of point c
    
    # Returns:
    Bool: True if point a, b, and c are ordered in in ccw way
    """
    # This is the determinant
    # ~3.5x faster than _old_ccw
    return ax*(by-cy) - bx*(ay-cy) + cx*(ay-by) > 0
    # See https://stackoverflow.com/questions/9455298/ for a faster one


def _test_ccw(iterations=100):
    for _ in range(iterations):
        ax, ay, bx, by, cx, cy = 100*(np.random.random(6) - .5)
        ccw1 = _old_ccw(ax, ay, bx, by, cx, cy)
        ccw2 = _ccw(ax, ay, bx, by, cx, cy)
        assert(ccw1 == ccw2)

def _time_ccw(iterations = 10**6, f = _ccw):
    start = time()
    for _ in range(iterations):
        ax, ay, bx, by, cx, cy = 100*(np.random.random(6) - .5)
        ccw1 = f(ax, ay, bx, by, cx, cy)
    end = time()
    return end-start

def _membershiptest(px, py, Edges, nely, nelx, ccw_cda):
    """
    Test whether the point p is insider the region of the hole. If p is inside the hole region,
    return True.
    
    # Arguments:
    px: float, the x coordinate of point p
    py: float, the y coordinate of point p
    Edges: nx5 float array, the edge set. In each row, [1,x1,y1,x2,y2] where 1 denotes the edge is a line segment, x1, y1, x2
    and y2 are the coordinates of the end points of the edge.
    nely: int, number of elements in y direction
    nelx: int, number of elements in x dirction
    ccw_cda: the results of _ccw(c,d,a)
    #bigNumbery: the y coordinate of some point that is out of the hole boundary.
    
    # Returns:
    Bool: True is p is inside the hole regin
    """
    #number_edges = len(Edges)
    crossNumber = 0
    ay = CONFIG['length'] + 10  #i.e. 30

    for i in range(len(Edges)):
        if Edges[i][0] == 1:
            cx = 0.05 * nelx * Edges[i][1]
            cy = 0.1 * nely * Edges[i][2]
            dx = 0.05 * nelx * Edges[i][3]
            dy = 0.1 * nely * Edges[i][4]
            
            crossNumber += (_ccw(1, ay, px, py, cx, cy) != _ccw(1, ay, px, py, dx, dy)) & \
               (ccw_cda[i] != _ccw(cx, cy, dx, dy, px, py))

    return crossNumber % 2 == 1


# Global _FEM things that never change.
# todo: Bake these constants in to avoid costly name lookups?
Emin = 10**-6
E0 = 10**11
nu = 0.3
ay = CONFIG['length'] + 10  #i.e. 30

A11 = np.array([[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]])
A12 = np.array([[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]])
B11 = np.array([[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]])
B12 = np.array([[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]])

A12T = np.transpose(A12)
B12T = np.transpose(B12)

A1 = np.concatenate((A11, A12), axis=1)
A2 = np.concatenate((A12T, A11), axis=1)
A = np.concatenate((A1, A2))

B1 = np.concatenate((B11, B12), axis=1)
B2 = np.concatenate((B12T, B11), axis=1)
B = np.concatenate((B1, B2))

KE = (1/(0.91*24))*(A+nu*B)


def _FEM(Edges, nely, nelx, image):
    """
    Finite element analysis of the bridge
    #inputs:
        Edges: get the Edges from circular packing
        nely: int, number of elements in y direction
        nelx: int, number of elements in x dorection
        image: True or False, True for drawing the bridge 
    #Aurguments:
        E0: constant, Young modules
        nu: const, Poisson modules
        fmag: int, magnitude of the load
        e: vector of elements of the hole
        KE: stiffness matrix of the element
        edofVec: vector of degrees of freedom of the elements
        edofMat: matrix of degrees of freedom for elements (each rows is for one element)
        ik,jk,sk: vectors for building the assembled stiffness matrix
        F: vector of forces
        U: displacement vector
        fixeddofs: vector of fixed degrees of freedom (boundary conditions)
        alldofs: all degrees of freedom
        freedofs: alldofs-fixeddofs
        K: global stiffness matrix
        A: area of each elements
        t: thickness of elements
        sigma: stress
    #outputs:
        Stress(MPa)
        area(m^2) 
    """
    # TODO: More optimizations; replace constants with static values, avoid loads.
    fmag = 10**7/(nelx)
    #Edges, r_ld, r = generate_boundary_edges()
    x = np.ones([nely, nelx])
    e = []
    # calculate the ccw(cx,cy,dx,dy,ax,ay)
    ccw_cda = np.ones(len(Edges))
    for i in range(len(Edges)):
        #ax = 0
        #ay = 30
        #cx = Edges[i][1]
        #cy = Edges[i][2]
        #dx = Edges[i][3]
        #dy = Edges[i][4]
        ccw_cda[i] = _ccw(Edges[i][1], Edges[i][2], Edges[i][3], Edges[i][4], 0, ay)
        
    for ely in range(nely):
        for elx in range(nelx):
            if _membershiptest((elx+1), (nely-ely-1),Edges,nely,nelx,ccw_cda) == True:
                x[ely][elx] = 0
                e.append(elx*nely + ely+1)
   
    #e = np.sort(e) #np.unique already sorts it all
    e = np.unique(e)

    a=np.linspace(1,(1+nelx)*(1+nely),(1+nelx)*(1+nely))
    
    bb = np.reshape(a, (nelx+1, nely+1))
    nodenrs = np.transpose(bb)
    
    b = np.zeros((nely, nelx))
    for i in range(nely):
        for j in range(nelx):
            b[i][j] = nodenrs[i][j]
    
    edofVec = np.reshape(2*b+1, ((nelx)*(nely), 1))
    edofVec = np.sort(edofVec, axis=0)
    
    edofMat = np.tile(edofVec, (1, 8))+np.tile(
            [0, 1, 2*nely+2, 2*nely+3, 2*nely, 2*nely+1, -2, -1], (nelx*nely, 1))
    
    
    iK = np.reshape(np.kron(edofMat, np.ones([8, 1])), (64*nelx*nely, 1))
    jK = np.reshape(np.kron(edofMat, np.ones([1, 8])), (64*nelx*nely, 1))
    
    xT = np.reshape(np.transpose(x), (1, nelx*nely))
    
    KE1 = np.reshape(np.transpose(KE),(np.size(KE),1))
    sK = np.reshape(np.transpose(np.dot(KE1,(Emin+np.power(xT,3)*(E0-Emin)))),(64*nelx*nely,1));
    
    #KE1 = np.reshape(np.transpose(KE),(np.size(KE),1))
    #sK = np.reshape(np.transpose(np.dot(KE1,(np.power(xT,1)))),(64*nelx*nely,1))
    F = np.zeros([2*(nely+1)*(nelx+1), 1])
    for i in range(nelx+1):
        F[2+2*(nely+1)*i-1][0] = -fmag
    
    U = np.zeros([2*(nely+1)*(nelx+1), 1])
    ##### Define the boundary conditions
    fixedcon11 = np.linspace(2*(nely+1),2*(nely+1)*(nelx+1),(nelx+1))
    fixedcon12 = np.linspace(2*(nely+1)-1,2*(nely+1)*(nelx+1)-1,(nelx+1))
    fixedcon1 = np.union1d(fixedcon11,fixedcon12)
    fixedcon2 = np.linspace(1,2*(nely+1)-1,(nely+1))
    
    
    fixeddofs = np.union1d(fixedcon1, fixedcon2)
    alldofs = np.linspace(1,2*(nely+1)*(nelx+1),2*(nely+1)*(nelx+1))  
    freedofs = np.setdiff1d(alldofs, fixeddofs)
    
    #### Define Global stiffness matrix
    sk = np.reshape(sK,(np.size(sK)))
    jk = np.reshape(jK,(np.size(jK)))-1
    ik = np.reshape(iK,(np.size(iK)))-1
    K = coo_matrix((sk,(ik,jk)),shape=(np.int(np.max(iK)),np.int(np.max(jK)))).tocsr()
    K=K.toarray()
    K = (K+np.transpose(K))/2
    
    K2 = np.zeros((len(freedofs),len(freedofs)))
    F2 = np.zeros((len(freedofs),1))
    for i in range(0,len(freedofs)):
        for j in range(0,len(freedofs)):
            K2[i][j] = K[np.int(freedofs[i])-1][np.int(freedofs[j])-1]
            F2[i][0] = F[np.int(freedofs[i])-1][0]   
        j=0
     
    #### Solving the equation using CG    
    U2 =cg(K2, F2, x0=None, tol=1e-05, maxiter=2000)               
    
    for i in range(0,len(freedofs)):    
        U[np.int(freedofs[i])-1] = U2[0][i]
    
    compliance = 0
    for elx in range(0, nelx):
        for ely in range(0, nely):
            n1 = (nely+1)*elx + ely
            n2 = (nely+1)*(elx+1) + ely
            edof = np.array([ 2*n1, 
                                   2*n1+1,
                                   2*n2,
                                   2*n2+1,
                                   2*n2+2,
                                   2*n2+3,
                                   2*n1+2,
                                   2*n1+3])
            edof = edof.transpose()
            Ue = U[edof]
            UeT = Ue.transpose()
            compliance = compliance + x[ely,elx]*np.dot(UeT, np.dot(KE, Ue))
    compliance = compliance[0][0]
    
    area = ((nelx*nely)-len(e))*A
    if  image:
        ax = plt.imshow(x)
        plt.show()
    return compliance,area

