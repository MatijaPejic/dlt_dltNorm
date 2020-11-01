import numpy as np
import math
#Example input
np.set_printoptions(suppress=True)
np.set_printoptions(precision=6)
a = np.array([-3, 2, 1])
b = np.array([-2, 5, 2])
c = np.array([1, 0, 3])
d = np.array([-7, 3, 1])
e = np.array([2, 1, 2])
f = np.array([-1, 2, 1])
g = np.array([1, 1, 1])
a1 = np.array([11, -12, 7])
b1 = np.array([25, -8, 9])
c1 = np.array([15, 4, 17])
d1 = np.array([14, -28, 10])
e1 = np.array([13, 8, 9])
f1 = np.array([11, -4, 5])
g1 = np.array([8, 4, 4])
p1 = np.array([a, b, c, d, e, f, g])
p2 = np.array([a1, b1, c1, d1, e1, f1, g1])

#Creating the needed 2D and 3D arrays
M3d = []
for point in p1:
    M3d.append(point / point[2])
M3d = np.array(M3d)

Mp3d = []
for point in p2:
    Mp3d.append(point / point[2])
Mp3d = np.array(Mp3d)

Mp2d = []
for point in Mp3d:
    Mp2d.append(point[:2])
Mp2d = np.array(Mp2d)

M2d = []
for point in M3d:
    M2d.append(point[:2])
M2d = np.array(M2d)


def DLT(p1, p2):
    A = []
    for pair in zip(p1, p2):
        M = pair[0]
        M_p = pair[1]
        x1 = M[0]
        x2 = M[1]
        x3 = M[2]
        x1_p = M_p[0]
        x2_p = M_p[1]
        x3_p = M_p[2]

        row1 = [0, 0, 0, -x3_p * x1, -x3_p * x2, -x3_p * x3, x2_p * x1, x2_p * x2, x2_p * x3]
        row2 = [x3_p * x1, x3_p * x2, x3_p * x3, 0, 0, 0, -x1_p * x1, -x1_p * x2, -x1_p * x3]
        A.append(row1)
        A.append(row2)
    u, s, v = np.linalg.svd(A)
    res = np.array_split(v[-1], 3)
    x = []
    for ar in res:
        y = []
        for num in ar:
            y.append(num)
        x.append(y)
    return np.array(x)


def Normalization(x):
    sum = 0
    m = np.mean(x, 0)
    for point in x:
        sum += np.linalg.norm(point - m)
    avg = sum / len(x)
    G = np.array([[1, 0, -m[0]],
                  [0, 1, -m[1]],
                  [0, 0, 1]
                  ])
    S = np.array([[math.sqrt(2) / avg, 0, 0],
                  [0, math.sqrt(2) / avg, 0],
                  [0, 0, 1]
                  ])
    Tr = np.dot(S, G)

    transformed_matrix = []
    x = np.insert(x, 2, 1, axis=1)
    for point in x:
        transformed_matrix.append(np.dot(Tr, point.T))
    transformed_matrix = np.array(transformed_matrix)

    return Tr, transformed_matrix

#DLT with Normalization 
T, M_nad = Normalization(M2d)
Tp, Mp_nad = Normalization(Mp2d)
P_nad = DLT(M_nad, Mp_nad)
P = np.dot(np.linalg.inv(Tp), np.dot(P_nad, T))
print("Transoform matrix:")
print(P)
