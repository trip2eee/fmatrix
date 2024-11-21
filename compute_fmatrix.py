import numpy as np
import cv2

class ComputeFMatrix:
    def __init__(self):
        pass

    
    def normalize_points(self, uv):

        s = 0
        u_mean = uv[:,0].mean()
        v_mean = uv[:,1].mean()

        N = len(uv)
        for ui, vi in uv:
            s += np.sqrt( (ui - u_mean)**2 + (vi - v_mean)**2 )
        
        s /= N*np.sqrt(2)

        uv_norm = np.zeros_like(uv, dtype=np.float64)

        uv_norm[:,0] = (uv[:,0] - u_mean) / s
        uv_norm[:,1] = (uv[:,1] - v_mean) / s
        
        T = np.array([
            [1/s,     0, -u_mean/s],
            [  0,   1/s, -v_mean/s],
            [  0,     0,         1]
        ], dtype=np.float64)

        return uv_norm, T

    def compute_fmatrix0(self, uv1, uv2):
        
        num_points = len(uv1)
        A = np.zeros([num_points, 9], dtype=np.float64)

        for i in range(num_points):
            
            u1 = uv1[i,0]
            v1 = uv1[i,1]
            
            u2 = uv2[i,0]
            v2 = uv2[i,1]

            A[i,0] = u1*u2
            A[i,1] = u2*v1
            A[i,2] = u2
            A[i,3] = u1*v2
            A[i,4] = v1*v2
            A[i,5] = v2
            A[i,6] = u1
            A[i,7] = v1
            A[i,8] = 1.0

        U, d, V = np.linalg.svd(A)
        V = V.T

        f = V[:,-1]

        F = f.reshape([3,3])
        
        U, d, V = np.linalg.svd(F)
        D = np.diag(d)

        # make F of rank 2
        D[2,2] = 0
        F = np.matmul(np.matmul(U, D), V)

        return F
    
    def optimize_fmatrix(self, uv1, uv2):
        pass
    
    def compute_fmatrix(self, uv1, uv2):
        
        uv1, T1 = self.normalize_points(uv1)
        uv2, T2 = self.normalize_points(uv2)

        print('normalized points')
        print(uv1[:,0].mean())
        print(uv1[:,1].mean())
        print(np.sqrt((uv1[:,0]**2 + uv1[:,1]**2)).mean())

        print(uv2[:,0].mean())
        print(uv2[:,1].mean())
        print(np.sqrt((uv2[:,0]**2 + uv2[:,1]**2)).mean())

        idx = np.arange(len(uv1))
        # idx = np.random.permutation(idx)
        idx = idx[:8]

        F0 = self.compute_fmatrix0(uv1[idx,:], uv2[idx,:])
        F0 = np.matmul(T2.T, np.matmul(F0, T1))

        return F0

