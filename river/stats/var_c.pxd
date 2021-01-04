# cython: boundscheck=False

from .mean_c cimport MeanC


cdef class VarC:
    r"""Running variance using Welford's algorithm, the Cython implementation.
    """

    cdef:
        double ddof, sigma
        MeanC mean

    cpdef void update(self, double x, double w)
    cpdef double get(self)
    cpdef void iadd(self, VarC other)
    cpdef void isub(self, VarC other)
    cpdef void copy(self, VarC other)
