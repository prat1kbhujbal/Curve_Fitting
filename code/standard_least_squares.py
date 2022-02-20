#!/usr/bin/env python

import pandas as pd
import numpy as pb
import matplotlib.pyplot as plt

data1 = pd.read_csv('../data_files/data1.csv')
X1 = data1['x'].values
Y1 = -data1['y'].values

data2 = pd.read_csv('../data_files/data2.csv')
X2 = data2['x'].values
Y2 = -data2['y'].values


def slsm(X, Y):
    x2 = []
    xy = []
    x2y = []
    x3 = []
    x4 = []
    sum_y = 0.0
    n = len(X)
    sum_x = 0.0
    sum_x2 = 0.0
    sum_xy = 0.0
    sum_x3 = 0.0
    sum_x2y = 0.0
    sum_x4 = 0.0

    for i in range(len(X)):
        x2.append(float(X[i]**2))
        xy.append(float(X[i] * Y[i]))
        x2y.append(float(x2[i] * Y[i]))
        x3.append(float(X[i]**3))
        x4.append(float(X[i]**4))
        sum_y = sum_y + Y[i]
        sum_x = sum_x + X[i]
        sum_x2 = sum_x2 + x2[i]
        sum_xy = sum_xy + xy[i]
        sum_x3 = sum_x3 + x3[i]
        sum_x2y = sum_x2y + x2y[i]
        sum_x4 = sum_x4 + x4[i]

    a = pb.array([
        [n, sum_x, sum_x2],
        [sum_x, sum_x2, sum_x3],
        [sum_x2, sum_x3, sum_x4]])
    b = pb.array([sum_y, sum_xy, sum_x2y])
    solutions = pb.linalg.solve(a, b)
    a1 = solutions[0]
    a2 = solutions[1]
    a3 = solutions[2]

    y_fit = []
    for i in range(0, len(X)):
        y = a1 + (a2 * (X[i])) + (a3 * X[i]**2)
        y_fit.append(y)

    return y_fit


y_fit1 = slsm(X1, Y1)
y_fit2 = slsm(X2, Y2)

fig = plt.figure(figsize=(9, 5))
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Data 1')

plt.scatter(X1, Y1, c='red')
plt.plot(X1, y_fit1, 'black')

fig = plt.figure(figsize=(9, 5))
plt.xlabel('X Axis')
plt.ylabel('Y Axis')
plt.title('Data 2')

plt.scatter(X2, Y2, c='red')
plt.plot(X2, y_fit2, 'black')
plt.show()
