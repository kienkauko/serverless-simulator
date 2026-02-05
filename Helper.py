import math
from scipy.special import gamma as gamma_func
from scipy.optimize import fsolve


def weibull_params_from_mean_std(mean, std):
    """
    Calculate Weibull shape (k) and scale (lambda) parameters from mean and std.
    
    For Weibull distribution:
    mean = lambda * Gamma(1 + 1/k)
    variance = lambda^2 * [Gamma(1 + 2/k) - Gamma(1 + 1/k)^2]
    
    Returns:
        (shape, scale): Weibull parameters k and lambda
    """
    cv = std / mean  # Coefficient of variation
    
    def equations(k):
        g1 = gamma_func(1 + 1/k)
        g2 = gamma_func(1 + 2/k)
        return (g2 / g1**2 - 1) - cv**2
    
    # Solve for shape parameter k (initial guess = 1.0)
    k_solution = fsolve(equations, 1.0)[0]
    
    # Calculate scale parameter lambda
    scale = mean / gamma_func(1 + 1/k_solution)
    
    return k_solution, scale


def lognormal_params_from_mean_std(mean, std):
    """
    Calculate Lognormal mu and sigma parameters from mean and std.
    
    For Lognormal distribution:
    mean = exp(mu + sigma^2/2)
    variance = (exp(sigma^2) - 1) * exp(2*mu + sigma^2)
    
    Returns:
        (mu, sigma): Lognormal parameters
    """
    variance = std ** 2
    sigma_sq = math.log(1 + variance / mean**2)
    sigma = math.sqrt(sigma_sq)
    mu = math.log(mean) - sigma_sq / 2
    
    return mu, sigma
