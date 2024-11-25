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

# K
# [[750.   0. 480.]
#  [  0. 750. 270.]
#  [  0.   0.   1.]]

# R
# [[ 0.99558784 -0.08774923 -0.03324032]
#  [ 0.08710265  0.99598989 -0.02042722]
#  [ 0.0348995   0.01744177  0.99923861]]

# t
# [[-2.]
#  [-1.]
#  [ 5.]]

uv1, uv2 = generate_points(K, R, t)

# convert points into the canonical form
np.random.seed(123)
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


