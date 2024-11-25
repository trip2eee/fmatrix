import numpy as np

def compute_K(fu, fv, width, height):
    # define K
    cu = width * 0.5
    cv = height * 0.5

    K = np.array([
        [fu, 0,  cu],
        [0,  fv, cv],
        [0,  0,  1]
    ], dtype=np.float64)

    return K

def compute_R(ax, ay, az):
    Rx = np.array([
        [1,          0,           0],
        [0, np.cos(ax), -np.sin(ax)],
        [0, np.sin(ax),  np.cos(ax)],
    ], dtype=np.float64)

    Ry = np.array([
        [ np.cos(ay), 0, np.sin(ay)],
        [          0, 1,          0],
        [-np.sin(ay), 0, np.cos(ay)],
    ], dtype=np.float64)

    Rz = np.array([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az),  np.cos(az), 0],
        [         0,           0, 1],
    ], dtype=np.float64)

    R = np.matmul(Rz, np.matmul(Ry, Rx))

    return R

def compute_t(tx, ty, tz):
    t = np.matrix([
        [tx],
        [ty],
        [tz],
    ], dtype=np.float64)

    return t

def generate_points(K, R, t, std_nosie=0):

    # Generate points
    uv1 = []
    uv2 = []

    rows = 5
    cols = 5    

    for i in range(rows):
        for j in range(cols):
            x = j - np.floor(cols/2)
            y = i - np.floor(rows/2)
            z = 10

            xyz0 = np.array([[x, y, z]]).T
            p0 = np.matmul(K, xyz0)
            
            u0 = p0[0,0] / p0[2,0] + np.random.randn()*std_nosie
            v0 = p0[1,0] / p0[2,0] + np.random.randn()*std_nosie
            uv1.append([u0, v0])

            xyz1 = np.matmul(R, xyz0) + t
            p1 = np.matmul(K, xyz1)

            u1 = p1[0,0] / p1[2,0] + np.random.randn()*std_nosie
            v1 = p1[1,0] / p1[2,0] + np.random.randn()*std_nosie
            
            uv2.append([u1, v1])

    uv1 = np.array(uv1, dtype=np.float64)
    uv2 = np.array(uv2, dtype=np.float64)

    return uv1, uv2

