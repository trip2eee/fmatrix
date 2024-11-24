import numpy as np
import cv2
import matplotlib.pyplot as plt

class ComputeFMatrix:
    def __init__(self):
        self.w_ortho = 1.0
        self.max_iter = 15

    
    def normalize_points(self, uv):

        s = 0
        u_mean = uv[:,0].mean()
        v_mean = uv[:,1].mean()

        N = len(uv)
        for i in range(N):
            ui = uv[i,0]
            vi = uv[i,1]
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
    
    def compute_X(self, P1, P2, uv1, uv2):
        N = len(uv1)
        X = np.zeros([N, 3], dtype=np.float64)

        p00, p01, p02, p03 = P2[0,:]
        p10, p11, p12, p13 = P2[1,:]
        p20, p21, p22, p23 = P2[2,:]

        for i in range(N):

            u1 = uv1[i,0]
            v1 = uv1[i,1]

            u2 = uv2[i,0]
            v2 = uv2[i,1]

            A = np.array([
                [1, 0, -u1],
                [0, 1, -v1],
                [p00-u2*p20, p01-u2*p21, p02-u1*p22],
                [p10-v2*p20, p11-v2*p21, p12-v1*p22],
            ], dtype=np.float64)

            b = np.array([
                [0],
                [0],
                [-p03+u2*p23],
                [-p13+v2*p23],
            ], dtype=np.float64)

            AtA = np.matmul(A.T, A)
            Atb = np.matmul(A.T, b)

            Xi = np.linalg.solve(AtA, Atb)

            X[i,:] = Xi[:,0]

            # Xi = np.append(Xi, 1).reshape([-1, 1])
            # proj1 = np.matmul(P1, Xi)
            # u_proj1 = proj1[0,0] / proj1[2,0]
            # v_proj1 = proj1[1,0] / proj1[2,0]

            # proj2 = np.matmul(P2, Xi)
            # u_proj2 = proj2[0,0] / proj2[2,0]
            # v_proj2 = proj2[1,0] / proj2[2,0]

            # d1 = (u1-u_proj1)**2 + (v1-v_proj1)**2
            # d2 = (u2-u_proj2)**2 + (v2-v_proj2)**2
            # print(d1)
            # print(d2)
            
        return X
    
    def compute_f_ortho(self, param):
        m00, m01, m02, tx = param[0:4, 0]
        m10, m11, m12, ty = param[4:8, 0]
        m20, m21, m22, tz = param[8:12, 0]

        f = (m00**2 + m10**2 + m20**2 - 1)**2 + \
            (m01**2 + m11**2 + m21**2 - 1)**2 + \
            (m02**2 + m12**2 + m22**2 - 1)**2 + \
            (m00*m01 + m10+m11 + m20*m21)**2 + \
            (m00*m02 + m10*m12 + m20*m22)**2 + \
            (m01*m02 + m11*m12 + m21*m22)**2
    
        return f

    def compute_J_ortho(self, param):
        m00, m01, m02, tx = param[0:4, 0]
        m10, m11, m12, ty = param[4:8, 0]
        m20, m21, m22, tz = param[8:12, 0]

        # J: 1 x (12 + 3n)
        num_unknowns = len(param)
        J = np.zeros([1, num_unknowns], dtype=np.float64)

        J[0,  0] = 4*m00*(m00**2 + m10**2 + m20**2 - 1) + 2*m01*(m00*m01 + m10*m11 + m20*m21) + 2*m02*(m00*m02 + m10*m12 + m20*m22)
        J[0,  1] = 2*m00*(m00*m01 + m10*m11 + m20*m21) + 4*m01*(m01**2 + m11**2 + m21**2 - 1) + 2*m02*(m01*m02 + m11*m12 + m21*m22)
        J[0,  2] = 2*m00*(m00*m02 + m10*m12 + m20*m22) + 2*m01*(m01*m02 + m11*m12 + m21*m22) + 4*m02*(m02**2 + m12**2 + m22**2 - 1)
        J[0,  3] = 0

        J[0,  4] = 4*m10*(m00**2 + m10**2 + m20**2 - 1) + 2*m11*(m00*m01 + m10*m11 + m20*m21) + 2*m12*(m00*m02 + m10*m12 + m20*m22)
        J[0,  5] = 2*m10*(m00*m01 + m10*m11 + m20*m21) + 4*m11*(m01**2 + m11**2 + m21**2 - 1) + 2*m12*(m01*m02 + m11*m12 + m21*m22)
        J[0,  6] = 2*m10*(m00*m02 + m10*m12 + m20*m22) + 2*m11*(m01*m02 + m11*m12 + m21*m22) + 4*m12*(m02**2 + m12**2 + m22**2 - 1)
        J[0,  7] = 0
        
        J[0,  8] = 4*m20*(m00**2 + m10**2 + m20**2 - 1) + 2*m21*(m00*m01 + m10*m11 + m20*m21) + 2*m22*(m00*m02 + m10*m12 + m20*m22)
        J[0,  9] = 2*m20*(m00*m01 + m10*m11 + m20*m21) + 4*m21*(m01**2 + m11**2 + m21**2 - 1) + 2*m22*(m01*m02 + m11*m12 + m21*m22)
        J[0, 10] = 2*m20*(m00*m02 + m10*m12 + m20*m22) + 2*m21*(m01*m02 + m11*m12 + m21*m22) + 4*m22*(m02**2 + m12**2 + m22**2 - 1)
        J[0, 11] = 0

        return J

    def compute_f_proj(self, uv1, uv2, param, i):
        m00, m01, m02, tx = param[0:4, 0]
        m10, m11, m12, ty = param[4:8, 0]
        m20, m21, m22, tz = param[8:12, 0]

        u1 = uv1[i,0]
        v1 = uv1[i,1]

        u2 = uv2[i,0]
        v2 = uv2[i,1]

        x = param[(12+(i*3))+0, 0]
        y = param[(12+(i*3))+1, 0]
        z = param[(12+(i*3))+2, 0]
        
        f = np.array([
            [u1 - x/z], 
            [v1 - y/z], 
            [u2 - (m00*x + m01*y + m02*z + tx)/(m20*x + m21*y + m22*z + tz)], 
            [v2 - (m10*x + m11*y + m12*z + ty)/(m20*x + m21*y + m22*z + tz)], 
            [m00**2 + m10**2 + m20**2 - 1], 
            [m01**2 + m11**2 + m21**2 - 1], 
            [m02**2 + m12**2 + m22**2 - 1], 
            [m00*m01 + m10*m11 + m20*m21], 
            [m00*m02 + m10*m12 + m20*m22], 
            [m01*m02 + m11*m12 + m21*m22]
        ], dtype=np.float64)

        return f
    
    def compute_J_proj(self, uv1, uv2, param, i):
        m00, m01, m02, tx = param[0:4, 0]
        m10, m11, m12, ty = param[4:8, 0]
        m20, m21, m22, tz = param[8:12, 0]

        u1 = uv1[i,0]
        v1 = uv1[i,1]

        u2 = uv2[i,0]
        v2 = uv2[i,1]

        x = param[(12+(i*3))+0, 0]
        y = param[(12+(i*3))+1, 0]
        z = param[(12+(i*3))+2, 0]
        
        # J: 10 x (12 + 3n)
        num_unknowns = len(param)
        J = np.zeros([10, num_unknowns], dtype=np.float64)

        J[0,12+(i*3)+0] = -1/z
        J[0,12+(i*3)+1] = 0
        J[0,12+(i*3)+2] = x/z**2

        J[1,12+(i*3)+0] = 0
        J[1,12+(i*3)+1] = -1/z
        J[1,12+(i*3)+2] = y/z**2


        J[2,:12] = np.array([-x/(m20*x + m21*y + m22*z + tz), -y/(m20*x + m21*y + m22*z + tz), -z/(m20*x + m21*y + m22*z + tz), -1/(m20*x + m21*y + m22*z + tz), 0, 0, 0, 0, x*(m00*x + m01*y + m02*z + tx)/(m20*x + m21*y + m22*z + tz)**2, y*(m00*x + m01*y + m02*z + tx)/(m20*x + m21*y + m22*z + tz)**2, z*(m00*x + m01*y + m02*z + tx)/(m20*x + m21*y + m22*z + tz)**2, (m00*x + m01*y + m02*z + tx)/(m20*x + m21*y + m22*z + tz)**2], dtype=np.float64)
        J[2,12+(i*3)+0] = -m00/(m20*x + m21*y + m22*z + tz) + m20*(m00*x + m01*y + m02*z + tx)/(m20*x + m21*y + m22*z + tz)**2        
        J[2,12+(i*3)+1] = -m01/(m20*x + m21*y + m22*z + tz) + m21*(m00*x + m01*y + m02*z + tx)/(m20*x + m21*y + m22*z + tz)**2
        J[2,12+(i*3)+2] = -m02/(m20*x + m21*y + m22*z + tz) + m22*(m00*x + m01*y + m02*z + tx)/(m20*x + m21*y + m22*z + tz)**2

        J[3,:12] = np.array([0, 0, 0, 0, -x/(m20*x + m21*y + m22*z + tz), -y/(m20*x + m21*y + m22*z + tz), -z/(m20*x + m21*y + m22*z + tz), -1/(m20*x + m21*y + m22*z + tz), x*(m10*x + m11*y + m12*z + ty)/(m20*x + m21*y + m22*z + tz)**2, y*(m10*x + m11*y + m12*z + ty)/(m20*x + m21*y + m22*z + tz)**2, z*(m10*x + m11*y + m12*z + ty)/(m20*x + m21*y + m22*z + tz)**2, (m10*x + m11*y + m12*z + ty)/(m20*x + m21*y + m22*z + tz)**2], dtype=np.float64)
        J[3,12+(i*3)+0] = -m10/(m20*x + m21*y + m22*z + tz) + m20*(m10*x + m11*y + m12*z + ty)/(m20*x + m21*y + m22*z + tz)**2
        J[3,12+(i*3)+1] = -m11/(m20*x + m21*y + m22*z + tz) + m21*(m10*x + m11*y + m12*z + ty)/(m20*x + m21*y + m22*z + tz)**2
        J[3,12+(i*3)+2] = -m12/(m20*x + m21*y + m22*z + tz) + m22*(m10*x + m11*y + m12*z + ty)/(m20*x + m21*y + m22*z + tz)**2

        J[4,:12] = np.array([2*m00, 0, 0, 0, 2*m10, 0, 0, 0, 2*m20, 0, 0, 0], dtype=np.float64)
        J[5,:12] = np.array([0, 2*m01, 0, 0, 0, 2*m11, 0, 0, 0, 2*m21, 0, 0], dtype=np.float64)
        J[6,:12] = np.array([0, 0, 2*m02, 0, 0, 0, 2*m12, 0, 0, 0, 2*m22, 0], dtype=np.float64)
        J[7,:12] = np.array([m01, m00, 0, 0, m11, m10, 0, 0, m21, m20, 0, 0], dtype=np.float64)
        J[8,:12] = np.array([m02, 0, m00, 0, m12, 0, m10, 0, m22, 0, m20, 0], dtype=np.float64)
        J[9,:12] = np.array([0, m02, m01, 0, 0, m12, m11, 0, 0, m22, m21, 0], dtype=np.float64)

        return J

    
    def compute_sum_f(self, uv1, uv2, param):
        # f = self.compute_f_ortho(param) * self.w_ortho
        sum_f = 0.0

        N = len(uv1)
        for i in range(N):
            f = self.compute_f_proj(uv1, uv2, param, i)
            sum_f += np.matmul(f.T, f)

        return sum_f / N
    
    def optimize_fmatrix(self, F0, P1, P2, uv1, uv2):
        
        X = self.compute_X(P1, P2, uv1, uv2)
        F = F0

        N = len(uv1)    # the number of points
        NUM_PARAM = 12 + (N*3)
        
        param = np.zeros([NUM_PARAM, 1])
        param[:12,0] = P2.reshape([-1,1])[:,0]
        param[12:,0] = X.reshape([-1,1])[:,0]
        
        sum_f = self.compute_sum_f(uv1, uv2, param)
        mu = 1e-3

        updateJ = True

        for iter in range(self.max_iter):

            if updateJ:
                # J = np.zeros([1, NUM_PARAM], dtype=np.float64)
                
                JtJ = np.zeros([NUM_PARAM, NUM_PARAM], dtype=np.float64)
                Jtf = np.zeros([NUM_PARAM, 1], dtype=np.float64)

                # J = self.compute_J_ortho(param) * self.w_ortho
                # JtJ += np.matmul(J.T, J)

                # f = self.compute_f_ortho(param) * self.w_ortho
                # Jtf += J.T * f

                for i in range(N):
                    J = self.compute_J_proj(uv1, uv2, param, i)
                    JtJ += np.matmul(J.T, J)

                    f = self.compute_f_proj(uv1, uv2, param, i)
                    Jtf += np.matmul(J.T, f)

                updateJ = False

            JtJ_temp = JtJ.copy()
            for i in range(NUM_PARAM):
                JtJ_temp[i,i] += mu * JtJ_temp[i,i]

            dp = -np.linalg.solve(JtJ_temp, Jtf)

            param_temp = param + dp
            sum_f_temp = self.compute_sum_f(uv1, uv2, param_temp)

            if sum_f_temp < sum_f:
                sum_f = sum_f_temp
                param = param_temp.copy()
                mu *= 0.1
                updateJ = True

            else:
                mu *= 10

            print('iter:{}, sum_f:{}, sum_f_temp:{}, mu:{}'.format(iter, sum_f, sum_f_temp, mu))

        X_opt = param[12:].reshape([-1, 3])

        plt.figure('X')
        plt.scatter(X[:,0], X[:,1], c='r')
        plt.scatter(X_opt[:,0], X_opt[:,1], c='b')

        P2 = param[:12].reshape([3,4])
        print('P2')
        print(P2)

        R = P2[:, :3]
        t = P2[:,3].reshape([3,1])
        
        t1 = t[0,0]
        t2 = t[1,0]
        t3 = t[2,0]

        tx = np.array([
            [0, -t3, t2],
            [t3, 0, -t1],
            [-t2, t1, 0]
        ], dtype=np.float64)

        F = np.matmul(tx, R)

        print('R')
        print(R)
        print(np.matmul(R[:,0].T, R[:,0]))
        print(np.matmul(R.T, R))

        print('rank(F)')
        print(np.linalg.matrix_rank(F))
        
        return F, X_opt

    
    def compute_fmatrix(self, uv1, uv2, K):
        
        # convert into canonical form
        # P = K[I | 0] -> [I | 0]
        # P' = K[M | m] -> [M | m]

        invK = np.linalg.inv(K)
        N = len(uv1)
        uv1 = np.hstack([uv1, np.ones([N, 1])])
        uv2 = np.hstack([uv2, np.ones([N, 1])])

        uv1 = np.matmul(invK, uv1.T).T
        uv2 = np.matmul(invK, uv2.T).T

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
        print(idx)

        F0 = self.compute_fmatrix0(uv1[idx,:], uv2[idx,:])
        F = F0

        U, d, V = np.linalg.svd(F)

        ep = U[:,-1].reshape([-1,1])
        
        e1 = ep[0,0]
        e2 = ep[1,0]
        e3 = ep[2,0]

        epx = np.array([
            [0, -e3, e2],
            [e3, 0, -e1],
            [-e2, e1, 0]
        ], dtype=np.float64)

        t = ep
        M = np.matmul(epx, F)

        print('M0')
        print(M)
        print(np.matmul(M.T, M))

        P1 = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ], dtype=np.float64)
        P2 = np.hstack([M, t])

        F, X = self.optimize_fmatrix(F, P1, P2, uv1, uv2)


        # decompose
        print('SVD')
        U, d, V = np.linalg.svd(F)
        D = np.diag(d)

        W = np.array([
            [0, -1, 0],
            [1,  0, 0],
            [0,  0, 1]
        ])
        Z = np.array([
            [ 0, 1, 0],
            [-1, 0, 0],
            [ 0, 0, 0]
        ])

        R2 = np.matmul(np.matmul(U, W), V)
        print('R2')
        print(R2)
        
        S = np.matmul(np.matmul(U, Z), U.T)
        tx = S[1,2]
        ty = -S[0,2]
        tz = S[0,1]

        t2 = np.array([[tx, ty, tz]]).T
        print('t2')
        print(t2)

        F = np.matmul(T2.T, np.matmul(F, T1))

        # convert back
        F = np.matmul(invK.T, np.matmul(F, invK))

        return F

