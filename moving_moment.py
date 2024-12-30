import numpy as np
from scipy.integrate import quad


def moving_moment(data, weight_func, radius, n):
    data = np.asarray(data)

    def weighted_moment(x):
        func_to_integrate = lambda t: ((t - x) ** n) * np.interp(t, np.arange(len(data)), data) * weight_func(t - x)
        moment_numerator, _ = quad(func_to_integrate, x - radius, x + radius)

        weight_to_integrate = lambda t: weight_func(t - x) ** n
        weight_integral, _ = quad(weight_to_integrate, x - radius, x + radius)
        normalization_factor = weight_integral ** n

        if normalization_factor != 0:
            return (moment_numerator / normalization_factor) ** (1/n)
        else:
            return np.nan

    moving_moments = [weighted_moment(x) for x in range(radius, len(data) - radius)]
    return np.array(moving_moments)
