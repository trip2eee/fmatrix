import numpy as np
import cv2
from utils import *
from compute_fmatrix import ComputeFMatrix
import matplotlib.pyplot as plt

# define K
scale = 1/4
width  = int(3840 * scale)
height = int(2160 * scale)

fu = 3000 * scale
fv = 3000 * scale

K = compute_K(fu, fv, width, height)
R = compute_R(1*np.pi/180, -2*np.pi/180, 5*np.pi/180)
# R = compute_R(0*np.pi/180, 0*np.pi/180, 0*np.pi/180)
t = compute_t(-2, -1, 5)
# t = compute_t(0, 0, 5)

print(K)
print(R)
print(t)

uv1, uv2 = generate_points(K, R, t)

# F, mask = cv2.findFundamentalMat(uv0, uv1, cv2.FM_8POINT, ransacReprojThreshold=3, confidence=0.99)
# print('F')
# print(F)


# E, mask = cv2.findEssentialMat(
#     points1=uv1,
#     points2=uv2,
#     cameraMatrix=K,
#     method=cv2.RANSAC,
#     prob=0.999,
#     threshold=1)

# # _, R2, t2, mask = cv2.recoverPose(E2, points1=uv0, points2=uv1, cameraMatrix=K, mask=mask)
# # print('R2')
# # print(R2)
# # print('t2')
# # print(t2)

# print('SVD')
# U, d, V = np.linalg.svd(E)
# D = np.diag(d)

# W = np.array([
#     [0, -1, 0],
#     [1,  0, 0],
#     [0,  0, 1]
# ])
# Z = np.array([
#     [ 0, 1, 0],
#     [-1, 0, 0],
#     [ 0, 0, 0]
# ])

# print('R2')
# R2 = np.matmul(np.matmul(U, W), V)
# print(R2)

# print('t2')
# S = np.matmul(np.matmul(U, Z), U.T)
# tx = S[1,2]
# ty = -S[0,2]
# tz = S[0,1]

# t2 = np.array([[tx, ty, tz]]).T
# print(t2)

# image1 = np.zeros([height, width, 3], dtype=np.uint8)
# image2 = np.zeros([height, width, 3], dtype=np.uint8)

# convert points into the canonical form

fmatrix = ComputeFMatrix()
F = fmatrix.compute_fmatrix(uv1, uv2, K)

print(F)

# draw epipolar line
epipolar = []
for i in range(len(uv1)):
    u = uv1[i,0]
    v = uv1[i,1]

    l0 = F[0,0]*u + F[0,1]*v + F[0,2]
    l1 = F[1,0]*u + F[1,1]*v + F[1,2]
    l2 = F[2,0]*u + F[2,1]*v + F[2,2]

    v20 = 0
    u20 = -l1/l0*v20 - l2/l0

    v21 = height
    u21 = -l1/l0*v21 - l2/l0

    epipolar.append([u20, u21, v20, v21])

epipolar = np.array(epipolar, dtype=np.float32)

plt.figure('image1')
plt.scatter(uv1[:,0], uv1[:,1], marker='o')
plt.axis('equal')

plt.axis((0, 960, 0, 480))
plt.gca().invert_yaxis()

plt.figure('image2')
for e in epipolar:
    plt.plot(e[0:2], e[2:4], c='r')
plt.scatter(uv2[:,0], uv2[:,1], marker='o')
plt.axis('equal')

plt.axis((0, 960, 0, 480))
plt.gca().invert_yaxis()

plt.show()


