from . import base
from .gaussianc import GaussianC

__all__ = ["Gaussian"]


class Gaussian(base.ContinuousDistribution):
    """Normal distribution with parameters mu and sigma.

    Examples
    --------

    >>> from river import proba

    >>> p = proba.Gaussian().update(6).update(7)

    >>> p
    𝒩(μ=6.500, σ=0.707)

    >>> p.pdf(6.5)
    0.564189

    """

    def __init__(self):
        # self._var = stats.Var()
        self._helper = GaussianC()

    @property
    def n_samples(self):
        # return self._var.mean.n
        return self._helper.mean_samples

    @property
    def mu(self):
        # return self._var.mean.get()
        return self._helper.mean

    @property
    def sigma(self):
        # return self._var.get() ** 0.5
        return self._helper.sigma

    @property
    def mode(self):
        return self._helper.mu

    def __str__(self):
        return f"𝒩(μ={self.mu:.3f}, σ={self.sigma:.3f})"

    def update(self, x, w=1.0):
        # self._var.update(x, w)
        self._helper.update(x, w)
        return self

    def pdf(self, x):
        return self._helper.pdf(x)

    def cdf(self, x):
        return self._helper.cdf(x)
