import numpy as np

np.set_printoptions(suppress=True)
np.set_printoptions(precision=5)
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


def DLP(p1, p2):
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


res = DLP(p1, p2)
print("Matrica preslikavanja:")
print(res)
