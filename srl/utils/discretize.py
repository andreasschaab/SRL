"""Turn continuous stochastic processes into finite grids + transition matrices.

Heterogeneous-agent models live on discrete grids, so we approximate each
continuous process by a finite Markov chain ``(grid, P)`` where ``P[i, j]`` is
the probability of moving from state ``i`` to state ``j``.

The two the tutorials use:
  * ``discrete_assets``: a geometrically-spaced asset/wealth grid (dense near
    the borrowing constraint, where policies bend most).
  * ``discrete_log_ar1``: a log-AR(1) income/productivity process (Tauchen-style
    binning).

``discretize_CIR`` / ``discretize_price_r_CIR`` build a mean-reverting (CIR)
interest-rate chain by the max-entropy moment-matching method of Farmer & Toda
(2017). No tutorial uses them; they are here for models that need a CIR price
process. Treat them as a black box unless you care how that chain is built.
"""
import jax.numpy as jnp
import numpy as np
from scipy.stats import gamma, norm
from scipy.optimize import minimize


def discrete_assets(amin, amax, N):
    """Geometrically-spaced asset grid on [amin, amax] with N nodes.

    The geometric spacing concentrates points near the borrowing constraint
    where policy functions are most curved.
    """
    pivot = np.maximum(np.abs(amin), 0.25)
    a_grid = np.geomspace(amin + pivot, amax + pivot, N) - pivot
    a_grid[0] = amin  # ensure exact endpoint
    return jnp.asarray(a_grid)


def discrete_log_ar1(rho, sigma, N):
    """Discretize log(Y') = rho * log(Y) + sqrt(1-rho^2) * sigma * e, e ~ N(0,1).

    Returns (Y_grid, P) where Y_grid is in levels (exp of the latent grid)
    and P is the N×N transition matrix.
    """
    assert 0 < rho < 1, "rho must be in (0,1)"
    assert sigma > 0, "sigma must be positive"
    assert N >= 3, "N must be at least 3"

    Y_grid = np.linspace(-3 * sigma, 3 * sigma, N)
    grid_size = Y_grid[1] - Y_grid[0]

    P_matrix = np.zeros((N, N))
    for i, Y_i in enumerate(Y_grid):
        mean = rho * Y_i
        std = sigma * np.sqrt(1 - rho**2)
        bin_edges = Y_grid - 0.5 * grid_size
        bin_edges = np.append(bin_edges, Y_grid[-1] + 0.5 * grid_size)
        probs = np.diff(norm.cdf(bin_edges, loc=mean, scale=std))
        probs = np.clip(probs, 1e-10, 1)
        probs /= probs.sum()
        P_matrix[i, :] = probs

    return jnp.exp(Y_grid), jnp.asarray(P_matrix)


def _cir_pdf(x, r, a, b, sigma, Delta):
    ea = np.exp(-a * Delta)
    cond_mean = r * ea + b * (1 - ea)
    cond_var = (sigma**2 / a) * (1 - ea) * (r * ea + b / 2)
    return norm.pdf(x, loc=cond_mean, scale=np.sqrt(cond_var))


def _discrete_approximation(X, moment_func, target_moments, q, constraints):
    """Max-entropy moment-matching discretization (helper for CIR)."""
    n = len(X)
    q_norm = q / np.sum(q)
    p0 = q_norm.copy()

    def objective(p):
        p_safe = np.clip(p, 1e-12, 1.0)
        entropy_term = np.sum(p_safe * np.log(p_safe / q_norm))
        moments = np.array([np.sum(p * moment_func(X)[i]) for i in range(len(target_moments))])
        moment_error = np.sum((moments - target_moments) ** 2) * 1e4
        return entropy_term + moment_error

    cons = [{"type": "eq", "fun": lambda p: np.sum(p) - 1}, *constraints]
    bounds = [(0, 1) for _ in range(n)]
    res = minimize(objective, p0, method="SLSQP", constraints=cons, bounds=bounds,
                   options={"maxiter": 1000, "ftol": 1e-10})
    p_opt = np.clip(res.x, 0, 1)
    p_opt /= p_opt.sum()
    final_moments = np.array([np.sum(p_opt * moment_func(X)[i]) for i in range(len(target_moments))])
    final_error = np.sum((final_moments - target_moments) ** 2)
    return p_opt, final_error, final_error < 1e-4


def discretize_CIR(a, b, sigma, Delta, N=11, coverage=0.99, method="even"):
    """Discretize a CIR process by max-entropy moment matching.

    See Farmer & Toda (2017) for the underlying method.
    """
    alpha = 2 * a * b / sigma**2
    beta = 2 * a / sigma**2

    p = np.array([(1 - coverage) / 2, (1 + coverage) / 2])
    quantiles = gamma.ppf(p, a=alpha, scale=1 / beta)

    if method == "even":
        X = np.linspace(quantiles[0], quantiles[1], N)
        W = np.ones_like(X)
    elif method == "exponential":
        X = np.exp(np.linspace(np.log(quantiles[0]), np.log(quantiles[1]), N))
        W = X.copy()
    else:
        raise ValueError(f"Unknown method '{method}', expected 'even' or 'exponential'.")

    W = W / np.sum(W)
    P = np.full((N, N), np.nan)
    scaling_factor = np.max(np.abs(X))
    ea = np.exp(-a * Delta)

    for ii in range(N):
        r = X[ii]
        cond_mean = r * ea + b * (1 - ea)
        cond_var = (sigma**2 / a) * (1 - ea) * (r * ea + b / 2)
        TBar = np.array([0, cond_var])
        q = W * _cir_pdf(X, r, a, b, sigma, Delta)

        def moment_func(x):
            x_centered = (x - cond_mean) / scaling_factor
            return np.vstack([x_centered, x_centered**2])

        target_moments = TBar / (scaling_factor ** np.array([1, 2]))
        p_opt, opt_value, ok = _discrete_approximation(X, moment_func, target_moments, q, constraints=[])
        if not ok or opt_value > 1e-4:
            # Second moment didn't match to tolerance; fall back to first moment only.
            p_opt, _, _ = _discrete_approximation(
                X,
                lambda x: np.array([(x - cond_mean) / scaling_factor]),
                np.array([0]),
                q,
                constraints=[],
            )
        P[ii, :] = p_opt

    P = P / P.sum(axis=1, keepdims=True)
    return X, P


def discretize_price_r_CIR(rho, sigma, mean, N):
    """Convenience wrapper: discretize a CIR price process given AR(1)-style params."""
    delta = 1
    a = -np.log(rho) / delta
    b = mean
    sigma_cont = sigma / np.sqrt(delta)
    cond = 2 * a * b - sigma_cont**2
    if cond <= 0:
        raise ValueError(f"Feller condition violated, 2ab - sigma^2 = {cond} must be positive.")
    P1, P1_trans = discretize_CIR(a, b, sigma_cont, delta, N, 0.99, "exponential")
    return jnp.asarray(P1), jnp.asarray(P1_trans)
