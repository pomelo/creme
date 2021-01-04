# cython: boundscheck=False


cdef class MeanC:
    """Running mean the Cython implementation.

    References
    ----------
    [^1]: [West, D. H. D. (1979). Updating mean and variance estimates: An improved method. Communications of the ACM, 22(9), 532-535.](https://people.xiph.org/~tterribe/tmp/homs/West79-_Updating_Mean_and_Variance_Estimates-_An_Improved_Method.pdf)
    [^2]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
    [^3]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)

    """

    cdef double n, mean

    cpdef void update(self, double x, double w)
    cpdef void revert(self, double x, double w)
    cpdef double get(self)
    cpdef double get_n(self)
    cpdef void iadd(self, MeanC other)
    cpdef void isub(self, MeanC other)
    cpdef void copy(self, MeanC other)
