"""
cython implementation of soft-DTW recursion.
Refs:
    - Blondel, M., Mensch, A., & Vert, J.-P. (2021).
    Differentiable Divergences Between Time Series. AISTATS. http://arxiv.org/abs/2010.08354
    - Mensch, A.,  Blondel, M. (2018).
    Differentiable Dynamic Programming for Structured Prediction and Attention. ICML. https://proceedings.mlr.press/v80/mensch18a.html

@author: Clément Lejeune <clementlej@gmail.com>
"""
cimport numpy as np
import numpy as np
cimport cython
from libc.math cimport exp, log

@cython.boundscheck(False)
cdef (double, double, double, double) _soft_min_argmin(
      double x, double y, double z) nogil:

    cdef double min_xyz = min(x, min(y, z))
    cdef double e_x
    cdef double e_y
    cdef double e_z
    cdef double nn
    cdef double soft_min

    e_x = exp(min_xyz - x)
    e_y = exp(min_xyz - y)
    e_z = exp(min_xyz - z)
    nn = e_x + e_y + e_z # normalizing constant
    soft_min = min_xyz - log(nn) # smoothed_min operator value

    return soft_min, e_x/nn, e_y/nn, e_z/nn


@cython.boundscheck(False)
@cython.wraparound(False) 
def _sdtw_C_cy(
    double[:, ::1] C,
    double gamma):

    dtype = np.double

    cdef Py_ssize_t i, j
    cdef Py_ssize_t len_X = C.shape[0]
    cdef Py_ssize_t len_Y = C.shape[0]
    cdef double big = 10**10
    
    cdef double [:] smarg = np.empty(4)
    cdef double[:, :] V = np.zeros((len_X + 1, len_Y + 1), dtype=dtype)
    cdef double[:, :, :] P = np.zeros((len_X + 2, len_Y + 2, 3), dtype=dtype)

    with nogil:
        
        # initilize first column to 1e10
        for i in range(1, len_X + 1):
            V[i, 0] = big

        # initialize firt row to 1e10
        for j in range(1, len_Y + 1):
            V[0, j] = big

        for i in range(1, len_X + 1):
            for j in range(1, len_Y + 1):
        
                soft_min, P[i, j, 0], P[i, j, 1], P[i, j, 2] = _soft_min_argmin(
                                                                    V[i, j - 1],
                                                                    V[i - 1, j - 1],
                                                                    V[i - 1, j])

                if gamma != 1.0:
                    V[i, j] = (C[i-1, j-1] / gamma) + soft_min
                else:
                    V[i, j] = C[i-1, j-1] + soft_min

    return gamma * V[len_X, len_Y]

