import numpy as np
from scipy.stats import poisson


class ConstantWeight:
    def __call__(self, t):
        return 1


class TriangleWeight:
    def __call__(self, t):
        return 1 - np.abs(t)


class GaussWeight:
    def __init__(self, sigma_):
        self.sigma_ = sigma_

    def __call__(self, t):
        return np.exp(-0.5 * (t / self.sigma_) ** 2) / (
            self.sigma_ * np.sqrt(2 * np.pi)
        )


class PoissonWeight:
    def __init__(self, lambda_):
        self.lambda_ = lambda_

    def __call__(self, t):
        adjusted_indices = np.abs(t) + self.lambda_ - np.max(np.abs(t))
        weights = poisson.pmf(adjusted_indices, mu=self.lambda_)
        weights /= np.sum(weights)
        return weights
