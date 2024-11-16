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

$T = \begin{bmatrix} 1/s & 0 & -\bar{u}/s \\ 1/s & 0 & -\bar{v}/s \\ 0 & 0 & 1\end{bmatrix}$

### compute_fmatrix0()
This method computes the initial fundamental matrix $\mathbf{F}_0$.

Algorithm 11.1 The normalized 8-point algorithm for $\mathbf{F}$ on page 282.

### optimize_fmatrix()


## Reference
- Multiple View Geometry in Computer Vision. R. Hartley, and A. Zisserman. Cambridge University Press, New York, NY, USA, 2 edition, (2003 )

