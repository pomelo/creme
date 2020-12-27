# cython: boundscheck=False


cdef class MeanC:
    """Running mean the Cython implementation.

    References
    ----------
    [^1]: [West, D. H. D. (1979). Updating mean and variance estimates: An improved method. Communications of the ACM, 22(9), 532-535.](https://people.xiph.org/~tterribe/tmp/homs/West79-_Updating_Mean_and_Variance_Estimates-_An_Improved_Method.pdf)
    [^2]: [Finch, T., 2009. Incremental calculation of weighted mean and variance. University of Cambridge, 4(11-5), pp.41-42.](https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)
    [^3]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)

    """

    def __init__(self):
        self.n = 0
        self.mean = 0.0

    cpdef void update(self, double x, double w):
        self.n += w
        if self.n > 0:
            self.mean += w * (x - self.mean) / self.n

    cpdef void revert(self, double x, double w):
        self.n -= w
        if self.n < 0:
            raise ValueError("Cannot go below 0")
        elif self.n == 0:
            self.mean = 0.0
        else:
            self.mean -= w * (x - self.mean) / self.n

    cpdef double get(self):
        return self.mean

    cpdef double get_n(self):
        return self.n

    cpdef void iadd(self, MeanC other):
        cdef double old_n = self.n
        self.n += other.n
        self.mean = (old_n * self.mean + other.n * other.mean) / self.n

    cpdef void isub(self, MeanC other):
        cdef double old_n = self.n
        self.n -= other.n

        if self.n > 0:
            self.mean = (old_n * self.mean - other.n * other.mean) / self.n
        else:
            self.n = 0.0
            self.mean = 0.0

    cpdef void copy(self, MeanC other):
        self.n = other.n
        self.mean = other.mean
