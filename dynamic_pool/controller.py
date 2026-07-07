"""Analytical idle-timeout controller for the dynamic serverless pool.

Implements the 3D-CTMC control idea without solving the full 3D chain.
Under the non-blocking assumption the warm pool X3 decouples from the
cold-starting (X1) and processing (X2) populations and is approximated by a
1D birth-death chain:

    birth  k -> k+1  at rate  lambda                (service completions
                                                     return warm containers)
    death  k -> k-1  at rate  lambda + k*theta      (k >= 1: warm-hit claims
                                                     plus idle reaping)

whose stationary distribution has the closed product form

    pi_k = pi_0 * prod_{j=1..k} lambda / (lambda + j*theta)

From it the two theta-dependent quantities of the paper's equations follow
directly: the warm-hit probability p_w(theta) = 1 - pi_0 (PASTA) and the
mean warm-pool size E[X3](theta) = sum_k k*pi_k.

On the birth rate: in the 3D chain the only inflow to X3 is the X2 -> X3
transition, whose instantaneous rate is X2(t)*mu, not lambda. The constant
birth rate lambda used here is justified in two steps. First, flow
conservation under non-blocking gives E[X2]*mu = lambda (every accepted
request completes exactly once). Second, and stronger: with non-blocking,
X2 acts as an infinite-server stage, and a Poisson(lambda) arrival stream
whose points are independently delayed (cold start + service) remains a
Poisson(lambda) process (displacement theorem), so the completion stream
feeding X3 is itself Poisson(lambda). What the 1D chain *does* neglect is
the correlation between claims and births (a warm-hit removes a container
now and returns it E[Bw] later through the same stream) and the fact that
the simulator reaps after a *deterministic* idle timeout t while the chain
assumes an exponential lifetime with matched mean t = 1/theta. Those two
points are where the residual model error lives.

Control problem (constrained form, not a weighted objective):

    minimize    E[R](theta) = lambda*(1 - p_w)*E[Bc]*r_t
                              + lambda*E[Bw]*r_a
                              + E[X3]*r_w
    subject to  E[Br](theta) = E[Bw] + (1 - p_w)*E[Bc]  <=  SLA

Since E[Br] <= SLA  <=>  p_w >= p_min := 1 - (SLA - E[Bw]) / E[Bc], and
p_w is increasing in the idle timeout t = 1/theta, the constraint carves
out t >= t_feasible. The solver grid-searches t in [t_min, t_max], keeps
the feasible points, and returns the cost-minimizing timeout. If even
t_max cannot meet the SLA the solver returns the most-feasible point
(largest p_w, i.e. t_max).
"""

from __future__ import annotations

import math
from functools import lru_cache


def warm_pool_stats(lam: float, theta: float,
                    k_max: int = 50_000, tol: float = 1e-12):
    """Stationary (p_w, E[X3]) of the 1D birth-death warm-pool chain.

    ``lam`` is the arrival rate (req/s), ``theta`` the reaping rate (1/s).
    The unnormalized product terms are accumulated until they fall below
    ``tol`` relative to the running total (or ``k_max`` is hit).
    """
    if lam <= 0.0:
        return 0.0, 0.0
    if theta <= 0.0:
        return 1.0, math.inf

    term = 1.0          # unnormalized pi_k, starting at pi_0 = 1
    total = 1.0         # sum of terms
    weighted = 0.0      # sum of k * terms
    k = 0
    while k < k_max:
        k += 1
        term *= lam / (lam + k * theta)
        total += term
        weighted += k * term
        if term < tol * total:
            break

    pi0 = 1.0 / total
    return 1.0 - pi0, weighted / total


def expected_latency(p_w: float, e_bw: float, e_bc: float) -> float:
    """E[Br] = E[Bw] + (1 - p_w) * E[Bc]   (Eq. 3d-process-time)."""
    return e_bw + (1.0 - p_w) * e_bc


def expected_resource(lam: float, p_w: float, e_x3: float,
                      e_bw: float, e_bc: float,
                      r_t: float, r_a: float, r_w: float) -> float:
    """E[R] per Eq. 3d-resource (Little's law on X1 and X2)."""
    return (lam * (1.0 - p_w) * e_bc * r_t
            + lam * e_bw * r_a
            + e_x3 * r_w)


@lru_cache(maxsize=4096)
def _solve_cached(lam: float, e_bw: float, e_bc: float,
                  r_t: float, r_a: float, r_w: float,
                  sla: float, t_min: float, t_max: float, t_step: float):
    # Required warm-hit probability from the SLA constraint. p_min >= 1
    # (SLA <= E[Bw]) makes every point infeasible -> fall back to t_max.
    p_min = 1.0 - (sla - e_bw) / e_bc

    best_t = None
    best_cost = math.inf
    fallback_t = t_min
    fallback_pw = -1.0

    n = max(1, int(round((t_max - t_min) / t_step)) + 1)
    for i in range(n):
        t = min(t_min + i * t_step, t_max)
        p_w, e_x3 = warm_pool_stats(lam, 1.0 / t)
        if p_w > fallback_pw:
            fallback_pw, fallback_t = p_w, t
        if p_w < p_min:
            continue
        cost = expected_resource(lam, p_w, e_x3, e_bw, e_bc, r_t, r_a, r_w)
        if cost < best_cost:
            best_cost, best_t = cost, t

    return best_t if best_t is not None else fallback_t


def solve_optimal_timeout(lam: float, e_bw: float, e_bc: float,
                          r_t: float, r_a: float, r_w: float,
                          sla: float, t_min: float, t_max: float,
                          t_step: float = 0.25) -> float:
    """Return the SLA-feasible, resource-minimal idle timeout (seconds).

    With no predicted traffic there is nothing to keep warm, so the pool is
    drained as fast as the actuator allows (t_min). Results are memoized on
    lambda rounded to 3 decimals (the remaining arguments are constant
    within a run), so repeated minutes with similar rates cost nothing.
    """
    if lam <= 0.0:
        return t_min
    return _solve_cached(round(lam, 3), e_bw, e_bc,
                         r_t, r_a, r_w, sla, t_min, t_max, t_step)
