# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 09:41:24 2019

@author: mmb18008
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix


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
    if np.linalg.det([[ax, bx, cx], [ay, by, cy], [1, 1, 1]]) > 0:
        return True
    else:
        return False


def _membershiptest(px, py, Edges, nely, nelx):
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
    
    # Returns:
    Bool: True is p is inside the hole regin
    """
    size_edges = np.size(Edges, axis=0)
    number_edges = size_edges
    crossNumber = 0
    bigNumberx = 1
    bigNumbery = 30
    ax = bigNumberx
    ay = bigNumbery
    bx = px
    by = py

    for i in range(0, number_edges):
        if Edges[i][0] == 1:
            cx = 0.05 * nelx * Edges[i][1]
            cy = 0.1 * nely * Edges[i][2]
            dx = 0.05 * nelx * Edges[i][3]
            dy = 0.1 * nely * Edges[i][4]
            if (_ccw(ax, ay, bx, by, cx, cy) != _ccw(
                    ax, ay, bx, by, dx, dy)) & (_ccw(cx, cy, dx, dy, ax, ay) !=
                                                _ccw(cx, cy, dx, dy, bx, by)):
                crossNumber = crossNumber + 1
    if np.mod(crossNumber, 2) == 1:
        return True
    elif np.mod(crossNumber, 2) == 0:
        return False


#Edges = [[1,14,0,12.0984,2.8021,0],
#         [1,12.0984,2.8021,11.0667,3.8240,0],
#         [1,11.0667,3.8240,9.975,4.9074,0],
#         [1,9.975,4.9074,8.4040,4.9074,0],
#         [1,8.4040,4.9015,6.5983,4.8969,0],
#         [1,6.5983,4.8978,4.6968,4.8969,0],
#         [1,4.6968,4.8969,2.6658,4.8931,0],
#         [1,2.6658,4.8931,0,4,8891,0]]
#if image=True , it plots the bridge



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
        Stress
        area 
    """
    E0 = 2e6                                  #young modules
    nu = 0.3                                  #poisson modules

    #nelx = 20;
    #nely = 10; 
    fmag = 10000 / (nelx)                        #magnitude of force 
    #Edges, r_ld, r = generate_boundary_edges()
    x = np.ones([nely, nelx])
    e = []

    for ely in range(0, nely):
        for elx in range(0, nelx):
            if _membershiptest((elx + 1), (nely - ely - 1), Edges, nely,
                               nelx) == True:
                x[ely][elx] = 0
                e.append(elx * nely + ely + 1)

    e = np.sort(e)
    e = np.unique(e)            #vector of removed elements

    A11 = [[12, 3, -6, -3], [3, 12, 3, 0], [-6, 3, 12, -3], [-3, 0, -3, 12]]
    A12 = [[-6, -3, 0, 3], [-3, -6, -3, -6], [0, -3, -6, 3], [3, -6, 3, -6]]
    B11 = [[-4, 3, -2, 9], [3, -4, -9, 4], [-2, -9, -4, -3], [9, 4, -3, -4]]
    B12 = [[2, -3, 4, -9], [-3, 2, 9, -2], [4, 9, 2, 3], [-9, -2, 3, 2]]
    A12T = np.transpose(A12)
    B12T = np.transpose(B12)

    A1 = np.concatenate((A11, A12), axis=1)
    A2 = np.concatenate((A12T, A11), axis=1)
    A = np.concatenate((A1, A2))

    B1 = np.concatenate((B11, B12), axis=1)
    B2 = np.concatenate((B12T, B11), axis=1)
    B = np.concatenate((B1, B2))
    KE = (1 / (0.91 * 24)) * (A + nu * B)                   #stiffness matrix of the element
    a = []
    for i in range(1, (1 + nelx) * (1 + nely) + 1):
        a.append(i)
    bb = np.reshape(a, (nelx + 1, nely + 1))
    nodenrs = np.transpose(bb)

    b = np.zeros((nely, nelx))
    for i in range(0, nely):
        for j in range(0, nelx):
            b[i][j] = nodenrs[i][j]

    edofVec = np.reshape(2 * b + 1, ((nelx) * (nely), 1))
    edofVec = np.sort(edofVec, axis=0)

    edofMat = np.tile(edofVec, (1, 8)) + np.tile(
        [0, 1, 2 * nely + 2, 2 * nely + 3, 2 * nely, 2 * nely + 1, -2, -1],
        (nelx * nely, 1))                                                       #matrix of degrees of freedom

    iK = np.reshape(np.kron(edofMat, np.ones([8, 1])), (64 * nelx * nely, 1))
    jK = np.reshape(np.kron(edofMat, np.ones([1, 8])), (64 * nelx * nely, 1))

    xT = np.multiply(np.reshape(np.transpose(x), (1, nelx * nely)), E0)
    m = 0
    n = 0
    KE1 = np.reshape(np.transpose(KE), (np.size(KE), 1))
    sK = np.reshape(
        np.transpose(np.dot(KE1, (np.power(xT, 1)))), (64 * nelx * nely, 1))

    for i in range(len(e) - 1, -1, -1):
        for j in range(64 * e[i] - 1, (64 * (e[i] - 1)) - 1, -1):
            iK = np.delete(iK, j, 0)
            jK = np.delete(jK, j, 0)
            sK = np.delete(sK, j, 0)
        edofMat = np.delete(edofMat, e[i] - 1, 0)

    F = np.zeros([2 * (nely + 1) * (nelx + 1), 1])              #force vector
    for i in range(0, nelx + 1):
        F[2 + 2 * (nely + 1) * i - 1][0] = -fmag

    for i in range(2 * (nely + 1) * (nelx + 1), -1, -1):
        c = np.isin(edofMat, i)
        if c.any() == True:
            continue
        else:
            np.delete(F, i - 1, 0)

    U = np.zeros([2 * (nely + 1) * (nelx + 1), 1])                      #displacement vector
    i = np.int(np.round(((nelx * nely) - np.max(e)) / 10))
    fixedcon = []

    for j in range(1, i + 1):
        fixedcon.append(2 * (nely + 1) * (np.round(np.max(e) / 10) + j + 1) - 1)
        fixedcon.append(2 * (nely + 1) * (np.round(np.max(e) / 10) + j))

    fixedcon2 = []
    for j in range(0, np.min(e) + 1):
        fixedcon2.append(2 * j)

    fixeddofs = np.union1d(fixedcon, fixedcon2)

    alldofs = []
    for i in range(1, 2 * (nely + 1) * (nelx + 1) + 1):
        alldofs.append(i)

    m = np.size(edofMat, axis=0)
    n = np.size(edofMat, axis=1)

    for i in range(0, 2 * (nely + 1) * (nelx + 1)):
        t = 0
        for j in range(0, m):
            for q in range(0, n):
                if edofMat[j][q] == alldofs[i]:
                    t = t + 1
                    break
            if t != 0:
                break
        if t == 0:
            alldofs[i] = 0

    for i in range(len(alldofs) - 1, -1, -1):
        if alldofs[i] == 0:
            alldofs = np.delete(alldofs, i)

    freedofs = np.setdiff1d(alldofs, fixeddofs)   #boundary conditions

    sk = np.reshape(sK, (np.size(sK)))
    jk = np.reshape(jK, (np.size(jK))) - 1
    ik = np.reshape(iK, (np.size(iK))) - 1
    K = coo_matrix(
        (sk, (ik, jk)), shape=(np.int(np.max(iK)), np.int(np.max(jK)))).tocsr()
    K = K.toarray()
    K = (K + np.transpose(K)) / 2                   #assembled stifness matrix

    K2 = np.zeros((len(freedofs), len(freedofs)))
    F2 = np.zeros((len(freedofs), 1))
    for i in range(0, len(freedofs)):
        for j in range(0, len(freedofs)):
            K2[i][j] = K[freedofs[i] - 1][freedofs[j] - 1]

            F2[i][0] = F[freedofs[i] - 1][0]
            j = 0

            #U2 = np.linalg.solve(F,K)
    U2 = np.dot(np.linalg.inv(K2), F2)
    for i in range(0, len(freedofs)):
        U[freedofs[i] - 1][:] = U2[i][0]                    #final displacement

    d = np.zeros([8, 1])
    sigma = np.zeros([np.size(edofMat, axis=0), 2])                 #stress
    A = 4000 / (nelx * nely)
    t = 50
    for i in range(0, np.size(edofMat, axis=0)):
        for j in range(0, 8):
            d[j][0] = U[np.int(edofMat[i][j]) - 1]

        sigma[i] = np.reshape(E0 / (2 * t * A * (1 - np.power(nu, 2))) * np.dot(
            [[1, nu], [nu, 1]],
            np.dot([[-1, 0, -1, 0, 1, 0, -1, 0], [0, -1, 0, -1, 0, 1, 0, 1]],
                   d)), (1, 2))
    sigma = abs(sigma)
    maxsigma = np.max(sigma)
    area = (nelx * nely) - len(e)
    if image:
        ax = plt.imshow(x)
        plt.show()
    return maxsigma, area

