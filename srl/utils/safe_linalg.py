"""JIT-safe linear-algebra helpers used by the SRL solvers.

Functions here are small and self-contained so they're easy to
read alongside the math in the tutorials. All are decorated with @jax.jit.
"""
import jax
import jax.numpy as jnp
from functools import partial


@jax.jit
def crra_util_func(c, sigma):
    """CRRA per-period utility, with log(c) at sigma=1.

    Floors consumption at 1e-5 to keep utility finite.
    """
    c = jnp.maximum(c, 1e-5)
    crra_u = jax.lax.cond(
        sigma == 1,
        lambda x: jnp.log(x),
        lambda x: (x ** (1 - sigma) - 1) / (1 - sigma),
        c,
    )
    return crra_u


@jax.jit
def interp_two_point_nonuniform(x, grid):
    """Linear-interpolation weights for a scalar x against a sorted, non-uniform grid.

    Returns (lo_idx, hi_idx, w_lo, w_hi) such that
        x ≈ w_lo * grid[lo_idx] + w_hi * grid[hi_idx],   w_lo + w_hi = 1.
    Saturates to 0/1 at the boundaries.
    """
    N = grid.shape[0]
    # rightmost i such that grid[i-1] <= x < grid[i]; clamp to [1, N-1]
    i_hi = jnp.clip(jnp.searchsorted(grid, x, side="right"), 1, N - 1).astype(jnp.int32)
    i_lo = (i_hi - 1).astype(jnp.int32)

    x_lo = grid[i_lo]
    x_hi = grid[i_hi]
    denom = jnp.maximum(x_hi - x_lo, 1e-12)  # guard against coincident nodes
    w_hi = jnp.clip((x - x_lo) / denom, 0.0, 1.0)
    w_lo = 1.0 - w_hi
    return i_lo, i_hi, w_lo, w_hi


@jax.jit
def find_clearing_point(B, Dt):
    """Find the two adjacent grid points on Dt straddling the target level B.

    Returns (lo_idx, hi_idx, w_lo, w_hi) such that
        B ≈ w_lo * Dt[lo_idx] + w_hi * Dt[hi_idx].
    Used inside `AUS_func` to clear the bond market at a price grid.
    """
    distances = jnp.abs(Dt - B)
    closest_idx = jnp.argmin(distances)
    closest_value = Dt[closest_idx]
    is_left = B < closest_value

    def left_case():
        i_hi = closest_idx
        left_mask = Dt < closest_value
        i_lo = jnp.argmin(jnp.where(left_mask, jnp.abs(Dt - B), jnp.inf))
        return i_lo, i_hi

    def right_case():
        i_lo = closest_idx
        right_mask = Dt > closest_value
        i_hi = jnp.argmin(jnp.where(right_mask, jnp.abs(Dt - B), jnp.inf))
        return i_lo, i_hi

    i_lo, i_hi = jax.lax.cond(is_left, left_case, right_case)
    i_lo, i_hi = jnp.minimum(i_lo, i_hi), jnp.maximum(i_lo, i_hi)

    Dt_lo = Dt[i_lo]
    Dt_hi = Dt[i_hi]
    denom = Dt_hi - Dt_lo
    safe_denom = jnp.where(denom == 0, 1.0, denom)

    w_hi = jnp.clip((B - Dt_lo) / safe_denom, 0.0, 1.0)
    w_lo = 1.0 - w_hi
    return i_lo, i_hi, w_lo, w_hi


@jax.jit
def apply_A_T(A, mt):
    """Apply A^T to the distribution vector mt, using a sparse triplet (rows, cols, vals).

    Computes m_{t+1}[col] = sum_{row} A[row, col] * m_t[row] via scatter-add,
    which is much cheaper than a dense J×J multiply when A is sparse.
    """
    rows, cols, vals = A
    return jnp.zeros_like(mt).at[cols].add(vals * mt[rows])


@partial(jax.jit, static_argnums=(1,))
def generate_uniform_m0(key, J):
    """Random initial distribution over J states (used during warm-up).

    Draws a monotone, non-uniform ramp between two random endpoints and
    normalizes. Gives every warm-up path a slightly different m0 so the
    solver doesn't overfit a single initial distribution.
    """
    key, _key = jax.random.split(key)
    x_left = jax.random.uniform(_key) + 1e-10
    x_right = jax.random.uniform(key) + 1e-10
    x = jnp.linspace(x_left, x_right, J)
    return x / x.sum()
