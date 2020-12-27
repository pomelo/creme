# cython: boundscheck=False


cdef class VarC:
    r"""Running variance using Welford's algorithm, the Cython implementation.

    Parameters
    ----------
    ddof
        Delta Degrees of Freedom. The divisor used in calculations is `n - ddof`, where `n`
        represents the number of seen elements.

    Attributes
    ----------
    mean : MeanC
        The Cython class for running mean.
    sigma : float
        The running variance.

    Notes
    -----
    The outcomes of the incremental and parallel updates are consistent with numpy's
    batch processing when $\\text{ddof} \le 1$.

    References
    ----------
    [^1]: [Wikipedia article on algorithms for calculating variance](https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Covariance)
    [^2]: [Chan, T.F., Golub, G.H. and LeVeque, R.J., 1983. Algorithms for computing the sample variance: Analysis and recommendations. The American Statistician, 37(3), pp.242-247.](https://amstat.tandfonline.com/doi/abs/10.1080/00031305.1983.10483115)

    """

    def __init__(self, ddof=1):
        self.ddof = ddof
        self.mean = MeanC()
        self.sigma = 0.0

    cpdef void update(self, double x, double w):
        cdef mean = self.mean.get()
        self.mean.update(x, w)
        if self.mean.n > self.ddof:
            self.sigma += (
                w * ((x - mean) * (x - self.mean.get()) - self.sigma) / (self.mean.n - self.ddof)
            )

    cpdef double get(self):
        return self.sigma

    @property
    def sigma(self):
        return self.sigma

    cpdef void iadd(self, VarC other):
        cdef:
            double old_n, delta

        if other.mean.n <= self.ddof:
            return

        old_n = self.mean.n
        delta = other.mean.get() - self.mean.get()

        # self.mean += other.mean
        self.mean.iadd(other.mean)

        # scale and merge sigma
        self.sigma = (old_n - self.ddof) * self.sigma + (other.mean.n - other.ddof) * other.sigma
        # apply correction
        self.sigma = (self.sigma + (delta * delta) * (old_n * other.mean.n) / self.mean.n) / (
            self.mean.n - self.ddof
        )

    cpdef void isub(self, VarC other):
        cdef:
            double old_n, delta

        old_n = self.mean.n
        delta = 0.0

        # self.mean -= other.mean
        self.mean.isub(other.mean)

        if self.mean.n > 0 and self.mean.n > self.ddof:
            delta = other.mean.get() - self.mean.get()
            # scale both sigma and take the difference
            self.sigma = (old_n - self.ddof) * self.sigma - (
                other.mean.n - other.ddof
            ) * other.sigma
            # apply the correction
            self.sigma = (self.sigma - (delta * delta) * (self.mean.n * other.mean.n) / old_n) / (
                self.mean.n - self.ddof
            )

        else:
            self.sigma = 0.0

    cpdef void copy(self, VarC other):
        self.ddof = other.ddof
        self.sigma = other.sigma
        self.mean.copy(other.mean)

    @property
    def mean_value(self):
        return self.mean.get()

    @property
    def mean_samples(self):
        return self.mean.n
