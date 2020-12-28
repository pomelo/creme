# -*- coding: utf-8 -*-
# cython: boundscheck=False, wraparound=False
# cython: linetrace=True

from libc cimport math
from ..stats.var_c cimport VarC

# tau = 2 * pi
DEF _tau = 6.283185307179586


cdef class GaussianC:
    """ Helper class for Gaussian in Cython.

    """

    cdef:
        VarC _var
        double _n_samples, _mu, _sigma, _variance

    def __init__(self):
        self._var = VarC()
        self._n_samples = 0.0
        self._mu = 0.0
        self._sigma = 0.0
        self._variance = 0.0

    cpdef void update(self, double x, double w):
        self._var.update(x, w)
        self._n_samples = self._var.mean.n
        self._mu = self._var.mean.get()
        self._variance = self._var.get()
        self._sigma = self._variance ** 0.5

    cpdef double pdf(self, double x):
        if self._variance:
            try:
                return math.exp((x - self._mu) ** 2 / (-2 * self._variance)) / math.sqrt(_tau * self._variance)
            except ValueError:
                return 0.
            except OverflowError:
                return 0.
        return 0.

    cpdef double cdf(self, double x):
        if self._sigma:
            try:
                return 0.5 * (1. + math.erf((x - self._mu) / (self._sigma * math.sqrt(2.))))
            except ZeroDivisionError:
                return 0.
        return 0.

    @property
    def mean_samples(self):
        return self._n_samples

    @property
    def mean(self):
        return self._mu

    @property
    def sigma(self):
        return self._sigma

    @property
    def mu(self):
        return self._mu
