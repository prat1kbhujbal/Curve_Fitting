#!/usr/bin/env python

import numpy as pb
import pandas as pd
import matplotlib.pyplot as plt
from numpy import linalg as LA

data = pd.read_csv(
    './data_files/linear_regression_dataset.csv')
X = data['age'].values
Y = data['charges'].values

x_n = (X - pb.min(X)) / (pb.max(X) - pb.min(X))
y_n = (Y - pb.min(Y)) / (pb.max(Y) - pb.min(Y))

X1 = pb.vstack((x_n, y_n))


def cov(x, y):
    xbar, ybar = x.mean(), y.mean()
    return pb.sum((x - xbar) * (y - ybar)) / (len(x) - 1)


def cov_mat(X):
    return pb.array([[cov(X[0], X[0]), cov(X[0], X[1])],
                     [cov(X[1], X[0]), cov(X[1], X[1])]])


covm = cov_mat(X1)
# print(covm)
eigenvalue, eigenvector = LA.eig(covm)
eigenvector_min = eigenvector[:, pb.argmin(eigenvalue)]
eigenvector_max = eigenvector[:, pb.argmax(eigenvalue)]


def llsm(X, Y):
    x = []
    y = []
    xy = []
    x2 = []
    n = len(X)
    sum_x = 0.0
    sum_x2 = 0.0
    sum_xy = 0.0
    sum_y = 0.0

    for i in range(len(X)):
        x.append(float(X[i]))
        xy.append(float(X[i] * Y[i]))
        x2.append(float(X[i] ** 2))
        sum_y = sum_y + Y[i]
        sum_x = sum_x + X[i]
        sum_x2 = sum_x2 + x2[i]
        sum_xy = sum_xy + xy[i]

    a = pb.array([
        [n, sum_x],
        [sum_x, sum_x2]])
    b = pb.array([sum_y, sum_xy])
    solutions = pb.linalg.solve(a, b)
    a1 = solutions[0]
    a2 = solutions[1]
    # print(solutions)
    return a2, a1


def tls(X, Y):
    x_mean = pb.mean(X)
    y_mean = pb.mean(Y)
    U = pb.vstack([X, Y]).T
    A = pb.dot(U.T, U)
    U, S_inv, V = SVD(A)
    a, b = V[:, V.shape[1] - 1]
    d = a * x_mean + b * y_mean
    a3 = -a / b
    a4 = d / b
    return a3, a4


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
    return U_eigenvectors, sigma, V_eigenvectors


def ransac(X, Y):
    req_error = 0
    prob_out = 200 / len(X)
    accuracy = 0.95
    iterations = pb.log(1 - accuracy) / pb.log(1 - pb.power((1 - prob_out), 2))
    iterations = pb.int(iterations)
    iterations = pb.maximum(iterations, 50)

    for i in range(iterations):
        random_data = pb.random.choice(len(X), size=2)
        x_rand = X[random_data]
        y_rand = Y[random_data]
        m, c = tls(x_rand, y_rand)
        error = Y - m * X - c
        error = error**2
        for i in range(len(error)):
            if float(error[i]) > 100:
                error[i] = 0
            else:
                error[i] = 1
        cal_error = pb.sum(error)
        if cal_error > req_error:
            req_error = cal_error
            coef = pb.array([m, c])
        if req_error / len(X) >= accuracy:
            break
    return coef


# Linear least square method
a2, a1 = llsm(X, Y)
y_llsm = []

for i in range(0, len(X)):
    y = a1 + (a2 * (X[i]))
    y_llsm.append(y)


# Total least square
a3, a4 = tls(X, Y)
y_tlsm = []
for i in range(0, len(X)):
    y = a4 + (a3 * (X[i]))
    y_tlsm.append(y)

# RANSAC
coef = ransac(X, Y)
y_ransac = []
for i in range(0, len(X)):
    y = coef[1] + (coef[0] * (X[i]))
    y_ransac.append(y)

plt.figure(1, figsize=(8, 4))
plt.scatter(X, Y, c='red')
plt.title("Eigen_values Representation")
plt.xlabel('Age')
plt.ylabel('Charges')
plt.quiver(
    pb.mean(X),
    pb.mean(Y),
    eigenvector_max[0] * (pb.max(X) - pb.min(X)) * pb.max(eigenvalue),
    eigenvector_max[1] * (pb.max(Y) - pb.min(Y)) * pb.max(eigenvalue),
    units='xy', angles='xy', scale_units='xy', scale=0.2)
plt.quiver(
    pb.mean(X),
    pb.mean(Y),
    eigenvector_min[0] * (pb.max(X) - pb.min(X)) * pb.min(eigenvalue),
    eigenvector_min[1] * (pb.max(Y) - pb.min(Y)) * pb.min(eigenvalue),
    units='xy', angles='xy', scale_units='xy', scale=0.2)

plt.figure(2, figsize=(8, 4))
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Linear least square')
plt.scatter(X, Y, c='red')
plt.plot(X, y_llsm, 'black')

plt.figure(3, figsize=(8, 4))
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('Total least square')
plt.scatter(X, Y, c='red')
plt.plot(X, y_tlsm, 'black')


plt.figure(4, figsize=(8, 4))
plt.xlabel('Age')
plt.ylabel('Charges')
plt.title('RANSAC')
plt.scatter(X, Y, c='red')
plt.plot(X, y_ransac, 'black')

plt.show()
