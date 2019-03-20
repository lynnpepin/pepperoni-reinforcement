# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 19:22:08 2019

@author: mehdi
maxstress,area = FEM(Edges,nely,nelx)
nely = number of elements in y axis(Height of the bridge)
nelx = number of elements in x axis(Length of the bridge)
you should bring Edges, r_ld, r = generate_boundary_edges() here, then run this code;
_ccw is used in _membershiptest;
"""
import math
import numpy as np
import matplotlib.pyplot as plt


def _ccw(ax, ay, bx, by, cx, cy):
    if np.linalg.det([[ax, bx, cx], [ay, by, cy], [1, 1, 1]]) > 0:
        return True
    else:
        return False


def _membershiptest(px, py, Edges,nely,nelx):
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
            cx =0.05*nelx* Edges[i][1]
            cy = 0.1*nely*Edges[i][2]
            dx =0.05*nelx* Edges[i][3]
            dy =0.1*nely* Edges[i][4]
            if (_ccw(ax, ay, bx, by, cx, cy) != _ccw(ax, ay, bx, by, dx, dy)) & (_ccw(cx, cy, dx, dy, ax, ay) != _ccw(cx, cy, dx, dy, bx, by)):
                crossNumber = crossNumber+1
    if np.mod(crossNumber, 2) == 1:
        return True
    elif np.mod(crossNumber, 2) == 0:
        return False


def _FEM(Edges,nely,nelx):
    E0 = 2e7
    nu = 0.3
    fmag = 10000/nelx
    #Edges, r_ld, r = generate_boundary_edges()
    x = np.ones([nely, nelx])
    e = []

    for elx in range(0, nelx):
        for ely in range(0, nely):
            if _membershiptest((elx), (nely-ely), Edges,nely,nelx) == True:
                x[ely][elx] = 0
                e.append(elx*nely + ely+1)

    e = np.sort(e)
    e = np.unique(e)

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
    KE = 1/(0.91*24)*(A+nu*B)
    a = []
    for i in range(1, (1+nelx)*(1+nely)+1):
        a.append(i)
    bb = np.reshape(a, (nelx+1, nely+1))
    nodenrs = np.transpose(bb)
    b = np.zeros((nely, nelx))
    for i in range(0, nely):
        for j in range(0, nelx):
            b[i][j] = nodenrs[i][j]

    edofVec = np.reshape(2*b+1, ((nelx)*(nely), 1))
    edofVec = np.sort(edofVec, axis=0)
    edofMat = np.tile(edofVec, (1, 8))+np.tile(
        [0, 1, 2*nely+2, 2*nely+3, 2*nely, 2*nely+1, -2, -1], (nelx*nely, 1))

    iK = np.reshape(np.kron(edofMat, np.ones([8, 1])), (64*nelx*nely, 1))
    jK = np.reshape(np.kron(edofMat, np.ones([1, 8])), (64*nelx*nely, 1))

    xT = np.multiply(np.reshape(np.transpose(x), (1, nelx*nely)), E0)
    m = 0
    n = 0
    KE1 = np.zeros([64, nelx*nely])

    for i in range(0, 8):
        for j in range(0, 8):
            KE1[n] = KE[j][i]*xT
            n = n+1

    sK = np.reshape(np.transpose(KE1), (64*nelx*nely, 1))

    for i in range(len(e)-1, -1, -1):
        for j in range(64*e[i], (64*(e[i]-1)), -1):
            iK = np.delete(iK, j, 0)
            jK = np.delete(jK, j, 0)
            sK = np.delete(sK, j, 0)
        edofMat = np.delete(edofMat, e[i], 0)

    F = np.zeros([2*(nely+1)*(nelx+1), 1])
    for i in range(0, nelx):
        F[2+2*(nely+1)*i][0] = -fmag

    for i in range(2*(nely+1)*(nelx+1), 0, -1):
        c = np.isin(edofMat, i)
        if c.any() == True:
            continue
        else:
            np.delete(F, i-1, 0)

    U = np.zeros([2*(nely+1)*(nelx+1), 1])
    i = np.int(np.round(((nelx*nely)-np.max(e))/10))
    fixedcon = []

    for j in range(1, i+1):
        fixedcon.append(2*(nely+1)*(np.round(np.max(e)/10)+j+1)-1)
        fixedcon.append(2*(nely+1)*(np.round(np.max(e)/10)+j+1))
    for i in range(1, nely+1):
        b = np.in1d(e, i)
        if b.any() == False:
            m = i
            break
        else:
            continue

    fixedcon2 = []
    for j in range(0, np.min(e)):
        fixedcon2.append(2*j)

    fixeddofs = np.union1d(fixedcon, fixedcon2)
    alldofs = []
    for i in range(1, 2*(nely+1)*(nelx+1)+1):
        alldofs.append(i)

    m = np.size(edofMat, axis=0)
    n = np.size(edofMat, axis=1)

    for i in range(0, 2*(nely+1)*(nelx+1)):
        t = 0
        for j in range(0, m):
            for q in range(0, n):
                if edofMat[j][q] == alldofs[i]:
                    t = t+1
                    break
            if t != 0:
                break
        if t == 0:
            alldofs[i] = 0

    for i in range(len(alldofs)-1, -1, -1):
        if alldofs[i] == 0:
            alldofs = np.delete(alldofs, i)

    freedofs = np.setdiff1d(alldofs, fixeddofs)

    for i in range(0, np.size(iK, axis=0)):
        for j in range(0, np.size(iK, axis=1)):
            iK[i][j] = np.int(iK[i][j])
            jK[i][j] = np.int(jK[i][j])

    for i in range(0, np.size(sK)):
        sK[i] = int(sK[i])

    K = np.zeros([np.int(np.max(iK)), np.int(np.max(jK))])
    for i in range(0, len(iK)):
        K[np.int(iK[i])-1][np.int(jK[i])-1] = sK[i]

    K = (K+np.transpose(K))/2
    c = np.linalg.lstsq(K, F)
    U = c[0]
    d = np.zeros([8, 1])
    sigma = np.zeros([np.size(edofMat, axis=0), 2])
    A = 4000/(nelx*nely)
    t = 5;
    for i in range(0, np.size(edofMat, axis=0)):
        for j in range(0, 8):
            d[j][0] = U[np.int(edofMat[i][j])-1]

        sigma[i] = np.reshape(E0/(2*t*A*(1-np.power(nu, 2))) * np.dot([[1, nu], [nu, 1]], np.dot(
            [[-1, 0, -1, 0, 1, 0, -1, 0], [0, -1, 0, -1, 0, 1, 0, 1]], d)), (1, 2))
    sigma = abs(sigma)
    maxsigma = np.max(sigma)
    area = (nelx*nely)-len(e)
    plt.imshow(x)
    return maxsigma, area
