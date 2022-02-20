#!/usr/bin/env python

import numpy as pb
from numpy import linalg as LA

x1, y1, xp1, yp1, x2, y2, xp2, yp2, x3, y3, xp3, yp3, x4, y4, xp4, yp4 = 5, 5, 100, 100, 150, 5, 200, 80, 150, 150, 220, 80, 5, 150, 100, 200

# Input matrix
A = pb.array([[-x1, -y1, -1, 0, 0, 0, x1 * xp1, y1 * xp1, xp1],
              [0, 0, 0, -x1, -y1, -1, x1 * yp1, y1 * yp1, yp1],
              [-x2, -y2, -1, 0, 0, 0, x2 * xp2, y2 * xp2, xp2],
              [0, 0, 0, -x2, -y2, -1, x2 * yp2, y2 * yp2, yp2],
              [-x3, -y3, -1, 0, 0, 0, x3 * xp3, y3 * xp3, xp3],
              [0, 0, 0, -x3, -y3, -1, x3 * yp3, y3 * yp3, yp3],
              [-x4, -y4, -1, 0, 0, 0, x4 * xp4, y4 * xp4, xp4],
              [0, 0, 0, -x4, -y4, -1, x4 * yp4, y4 * yp4, yp4]])


def SVD(A):
    AT = A.T
    AAT = A.dot(AT)
    U_eigenvalues, U_eigenvectors = LA.eig(AAT)
    decending_sort = U_eigenvalues.argsort()[::-1]
    U_eigenvalues[::-1].sort()
    U_eigenvectors = U_eigenvectors[:, decending_sort]

    ATA = AT.dot(A)
    V_eigenvalues, V_eigenvectors = LA.eig(ATA)
    decending_sort = V_eigenvalues.argsort()[::-1]
    V_eigenvalues[::-1].sort()
    # print(V_eigenvalues)
    V_eigenvectors = V_eigenvectors[:, decending_sort]
    VT_eigenvectors = V_eigenvectors.T
    diag_U = pb.diag((pb.sqrt(U_eigenvalues)))
    sigma = pb.zeros_like(A)
    sigma[:diag_U.shape[0], :diag_U.shape[1]] = diag_U

    H = V_eigenvectors[:, V_eigenvectors.shape[1] - 1]
    H = H.reshape([3, 3])
    H = H / H[2, 2]
    return U_eigenvectors, sigma, VT_eigenvectors, H


U, Sigma, VT, H = SVD(A)

print("U = ", U)
print("Sigma = ", Sigma)
print("VT = ", VT)
print("Homography Matrix = ", H)
