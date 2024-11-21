# Fundamental Matrix
Estimation of fundamental matrix

## ComputeFMatrix
### normalize_points()
This method normalizes input points as follows (isotropic scaling on page 107).
- centroid of points: (0, 0)
- average distance from the centroid: $\sqrt{2}$

$\bar{u} = \frac{1}{N}\sum_i{u_i} $

$\bar{v} = \frac{1}{N}\sum_i{v_i} $

$\frac{1}{N}\sum_i{\sqrt{\left(\frac{u_i - \bar{u}}{s}\right)^2 + \left(\frac{v_i - \bar{v}}{s}\right)^2}} = \sqrt{2}$

$s = \frac{1}{N\sqrt{2}}\sum_i{\sqrt{\left(u_i - \bar{u}\right)^2 + \left(v_i - \bar{v}\right)^2}}$

$T = \begin{bmatrix} 1/s & 0 & -\bar{u}/s \\ 0 & 1/s & -\bar{v}/s \\ 0 & 0 & 1\end{bmatrix}$

### compute_fmatrix0()
This method computes the initial fundamental matrix $\mathbf{F}_0$.

Algorithm 11.1 The normalized 8-point algorithm for $\mathbf{F}$ on page 282.

### optimize_fmatrix()

Algorithm 11.3. The Gold Standard algorithm for estimating F from image correspondences on page 285.

1. Compute an initial rank 2 estimate of $\hat{\mathbf{F}}$ using a linear algorithm - compute_fmatrix0().

2. Compute an initial estimate of the subsidiary variables $\lbrace \hat{\mathbf{x}}_i, \hat{\mathbf{x}}'_i\rbrace$ as follows:

    a. Choose camera matrices $\mathbf{P} = \begin{bmatrix} \mathbf{I} \mid \mathbf{0} \end{bmatrix}$ and $\mathbf{P}' = \begin{bmatrix} \mathbf{\left[\mathbf{e}' \right]_\times \hat{\mathbf{F}}} \mid \mathbf{e'} \end{bmatrix}$, where $\mathbf{e}'$ is the left nullspace of $\hat{\mathbf{F}}$ (Result A4.4).

    b. From the correspondence $x_i \leftrightarrow x'_i$ and $\hat{F}$ determine an estimate of $\hat{X}_i$ using the triangulation method.

    c. The correspondence consistent with $\hat{F}$ is obtained as $\hat{x}_i = P\hat{X}_i$, $\hat{x}'_i = P\hat{X}'_i$.

3. Minimize the cost $\sum_i d\left(x_i, \hat{x}_i\right)^2 + d\left(x'_i, \hat{x}'_i\right)^2 $ over $F$ and $\hat{X}_i$

   a. 3n + 12 variables, 3n for the n 3D points $\hat{X}_i$ and 12 for the camera matrix $P' = \begin{bmatrix}M \mid t\end{bmatrix}$.
   
   b. $\hat{F} = \left[t\right]_\times M$ and $\hat{x}_i = P\hat{X}_i$, $\hat{x}'_i = P'\hat{X}'_i$


- Result A4.3
  
  $\left[a\right]_\times = \left[a\right]_\times \left[a\right]_\times \left[a\right]_\times $ (up to scale)

  $\left[a\right]^3_\times = -\lVert a \rVert^2  \left[a\right]_\times $ (including scale)

- Result A4.4

  $F = \left[e'\right]_\times M$ is a fundamental matrix, then $\left[e'\right]_\times \left[e'\right]_\times F = F$ (up to scale).

  $\left[e'\right]_\times \left[e'\right]_\times \left[e'\right]_\times M = -\lVert e'\rVert \left[e'\right]_\times M = \left[e'\right]_\times M$ (up to scale).
  
  Hence one may decompose $F$ as $F = \left[e'\right]_\times M$, where $M = \left[e'\right]_\times F$.








## Reference
- Multiple View Geometry in Computer Vision. R. Hartley, and A. Zisserman. Cambridge University Press, New York, NY, USA, 2 edition, (2003 )

