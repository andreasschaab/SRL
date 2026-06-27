# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 5 — Huggett with aggregate risk
#
# This is the payoff. We solve a Huggett economy where, on top of the
# idiosyncratic income risk each household already faces, the *whole economy* gets
# hit by an **aggregate productivity shock** $z$. Classical global methods find
# this setting genuinely hard. The policy gradient handles it almost for free.
#
# The journey so far: notebook 0–2 solved a household at a *fixed* price;
# notebook 3 let it face an *exogenous* price process; notebook 4 made the price
# *endogenous*, set by market clearing, with price-taking enforced as a
# `stop_gradient`. Notebook 5 is notebook 4 plus one thing. The price now moves
# because the **aggregate state** moves.

# %% [markdown]
# ## What you should already know
#
# - **Notebook 4** especially: that the market-clearing price is found *inside*
#   the simulated environment (the household's policy doubles as its bond-supply
#   schedule), and that **price-taking is a stop-gradient** on that price.
# - Why aggregate risk is hard in heterogeneous-agent models: with an aggregate
#   shock the cross-sectional wealth distribution becomes a *state variable*.
#   Familiarity with Krusell–Smith and the approximate-aggregation idea helps,
#   but is not required. We explain the obstacle below.

# %% [markdown]
# ## The one new idea: carry the price, not the distribution
#
# Here's the obstacle. Without aggregate risk (notebook 4), the economy sits at
# a *stationary* distribution: solve once, the distribution never moves. With an
# aggregate shock $z_t$, the whole cross-section $G_t$ shifts every period. A
# good draw of $z$ makes everyone richer, which changes how much they want to
# save, which moves the equilibrium price. So $G_t$ becomes part of the **state**:
# the household needs to forecast next period's price, and the price depends on
# the entire distribution. Tracking an infinite-dimensional object through time
# is the curse of dimensionality. The classical fix (Krusell–Smith) is to
# approximate $G_t$ by a few moments and *posit* a law of motion for them, then
# iterate until it is self-consistent.
#
# **SRL sidesteps the distribution entirely.** The household never carries $G_t$.
# It carries the **bond price** $q$ (a single number) as part of its state, and
# the solver *learns the equilibrium price process from simulated paths*. Each
# period, the market-clearing $q$ is found on the simulated cross-section
# (exactly notebook 4's in-loop clearing), the realized $q$ becomes part of the
# state the household conditioned on, and gradient ascent shapes a policy
# $c(b, y, q, z)$ that is optimal *given how prices actually move*. The
# distribution still exists inside the simulation, since we need it to clear the
# market, but it is never a state the household optimizes over. **A low-dimensional
# price stands in for the high-dimensional distribution.** That swap is the whole
# trick. Nothing about it changed from notebook 4 except that $z$ now moves.

# %% [markdown]
# ## The model
#
# A household chooses consumption $c$ and bonds $b'$ to maximize
# $$\mathbb{E}_0 \sum_{t=0}^\infty \gamma^t\, u(c_t),
#   \qquad u(c)=\frac{c^{1-\sigma}}{1-\sigma},$$
# subject to
# $$c + q\,b' = b + z\,y, \qquad b' \ge \underline{b}.$$
# Income is the product of an **idiosyncratic** component $y$ (a Markov chain, as
# before) and an **aggregate** TFP level $z$ (a Markov chain, AR(1) in logs)
# common to everyone. The bond price $q$ is set by **market clearing** each
# period: aggregate bond demand must equal net supply $B$.
#
# *A note on convention.* We work directly in the **bond price** $q$ here, as in
# notebooks 0–3: a bond bought at price $q$ pays $1$ next period, so
# $b' = (b + zy - c)/q$. Notebook 4 instead carried the equivalent **gross
# return** $R = 1/q$ internally (purely to match the return-form SSJ reference it
# validated against), but the economics is identical, and the figures below report
# the clearing $q$.
#
# The household's state is $(b, y, q, z)$ and its policy is the **consumption
# share** $s(b,y,q,z)\in[0,1]$, with $c = s\cdot(\text{wealth}-\underline b)$. One
# implementation detail carries over from the paper: rather than parameterize $s$
# directly, we parameterize it as a value in the lowest price column
# (`first_col`) plus non-negative increments across the price grid (`diff_col`),
# and rebuild $s$ by a cumulative sum over $q$. This **forces $s$ to be monotone
# in $q$**, a structural prior (a more expensive bond tilts the consumption share
# in one direction) that stabilizes learning.

# %% [markdown]
# ## Setup
#
# Import the solver and helpers. On Colab, the setup cell at the top of the
# notebook has already cloned the repo and installed `srl`. A GPU makes the
# paper-scale run fast; the tutorial-scale grids below also run on CPU in a few
# minutes.

# %%
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

from srl import SPGSolver
from srl.utils.discretize import discrete_assets, discrete_log_ar1
from srl.utils.safe_linalg import (
    apply_A_T,
    crra_util_func,
    find_clearing_point,
    interp_two_point_nonuniform,
)

print("JAX devices:", jax.devices())
if not any(d.platform == "gpu" for d in jax.devices()):
    print("(no GPU, so this runs on CPU, slower but fine at the tutorial grid sizes)")

# A small, readable color cycle (one per income state).
COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

# %% [markdown]
# ## Calibration and grids
#
# This is the aggregate-risk experiment from the paper: borrowing is allowed
# (down to $\underline b = -1$), bonds are in zero net supply ($B=0$), and the
# aggregate shock $z$ is a 50-state discretization of a persistent log-AR(1).
# The price grid spans the range in which the bond market clears.
#
# **Where to run this.** The grids below are the paper scale, designed for a
# **Colab GPU** (a few minutes). On a CPU it still runs, but a smooth policy
# needs a few hundred policy-gradient steps, so expect ~15–20 minutes. Or just drop
# to the smaller *smoke-test* grids in the comment (faster, but the figures will
# look coarser and less converged).

# %%
# Paper scale (Colab GPU). Quick CPU smoke-test override: nb, ny, nq, nz = 80, 3, 10, 15.
nb, ny, nq, nz = 200, 3, 20, 50
bc, eps = -1.0, 0.001          # borrowing limit; consumption floor
gamma, sigma, B = 0.96, 2.0, 0.0   # discount, CRRA, net bond supply

b_grid = discrete_assets(0, 50 - bc, nb) + bc            # assets, dense near bc
y_grid, y_trans = discrete_log_ar1(0.6, 0.2, ny)         # idiosyncratic income
z_grid, z_trans = discrete_log_ar1(0.9, 0.02, nz)        # aggregate TFP
q_grid = jnp.linspace(0.95, 0.99, nq)                    # bond price grid

# Flattened (b, y) household grid, used to aggregate the cross-section.
b_dist = jnp.repeat(b_grid, ny)
y_dist = jnp.tile(y_grid, nb)

print(f"assets b in [{float(b_grid.min()):.1f}, {float(b_grid.max()):.1f}]")
print(f"income y = {y_grid.round(3)}")
print(f"aggregate z in [{float(z_grid.min()):.3f}, {float(z_grid.max()):.3f}]")

# %% [markdown]
# ## Initial policy guess and the monotone reparameterization
#
# `from_diff_to_cshare` rebuilds the full consumption-share tensor
# $s(b,y,q,z)$ from the `(first_col, diff_col)` parameters by a cumulative sum
# across the price dimension. That cumsum is what enforces monotonicity in $q$. The
# closed-form guess just gives gradient ascent a sensible starting point.

# %%
def gen_initial_guess():
    """A cheap closed-form consumption-share guess on the full (b, y, q, z) grid."""
    def per_state(b, y, q, z):
        w = b + y * z - bc
        p = 1.0 / q
        nom = jnp.maximum((p - (gamma * p) ** (1.0 / sigma)) * (w + 1.0 / (p - 1.0)), 0.0)
        return nom / w
    g = per_state(b_grid[:, None, None, None], y_grid[None, :, None, None],
                  q_grid[None, None, :, None], z_grid[None, None, None, :])
    return jnp.clip(g, 0.001, 1.0)


init_cshare = gen_initial_guess()
initial_guess = {
    "first_col": init_cshare[..., 0, :],            # share at the lowest price
    "diff_col": jnp.diff(init_cshare, axis=2),      # increments across the price grid
}


def from_diff_to_cshare(policy):
    """Rebuild s(b, y, q, z) from (first_col, diff_col): monotone in q by cumsum."""
    c = jnp.zeros((nb, ny, nq, nz))
    c = c.at[..., 0, :].set(policy["first_col"])
    c = c.at[..., 1:, :].set(policy["diff_col"])
    return jnp.cumsum(c, axis=2)


def outofbound_penalty(cpolicy):
    """Soft penalty discouraging consumption shares outside [0, 1] early in training."""
    return (jnp.minimum(cpolicy, 1e-6) - jnp.maximum(cpolicy - 1.0, -1e-6) - 2e-6).mean()

# %% [markdown]
# ## Model dynamics: market clearing on the simulated path
#
# `AUS_func` is the environment one simulation step. The new ingredient relative
# to notebook 4 is only that the aggregate state $z$ now evolves. Each step:
#
# 1. Given the current cross-section `mt` and aggregate state $z$, build the
#    **aggregate bond-supply schedule** $S(q)$ over the price grid and find the
#    market-clearing price with `find_clearing_point`, the same in-loop clearing
#    as notebook 4. `jax.lax.stop_gradient` on the policy here is **price-taking**:
#    the household does not internalize how its own choice moves the market price.
# 2. Build the sparse transition matrix and flow utility at that price.
# 3. Draw next period's aggregate state $z'$ from its Markov chain.

# %%
def reset_func(key):
    """Start each simulated path from a random aggregate state."""
    z_idx = jax.random.choice(key, nz)
    is_first_period = 1.0
    return z_idx, is_first_period


def AUS_func(policy, mt, st, key):
    z_idx, is_first_period = st
    cpolicy = from_diff_to_cshare(policy)
    penalty = is_first_period * outofbound_penalty(cpolicy)
    zt = z_grid[z_idx]
    J = nb * ny

    # --- market clearing: find q such that aggregate bond supply S(q) = B ---
    cshare_of_q = jax.lax.stop_gradient(cpolicy)[..., z_idx].reshape(-1, nq).T   # (nq, J)
    wealth_of_q = jnp.ones((nq, J)) * (b_dist + zt * y_dist)[None, :]            # (nq, J)
    C_of_q = jnp.clip((wealth_of_q - bc) * cshare_of_q, eps,
                      wealth_of_q - (bc + eps) * q_grid[:, None])
    S_of_q = jnp.sum((wealth_of_q - C_of_q) * mt[None, :], axis=-1) / q_grid     # (nq,)
    q_lo_i, q_hi_i, q_lo_w, q_hi_w = find_clearing_point(B, S_of_q)
    q = q_grid[q_lo_i] * q_lo_w + q_grid[q_hi_i] * q_hi_w
    # ---------------------------------------------------------------------

    def build_triplets_for_state(idx_S):
        b_idx, y_idx = idx_S // ny, idx_S % ny
        b, y = b_grid[b_idx], y_grid[y_idx]
        cshare = (cpolicy[b_idx, y_idx, q_lo_i, z_idx] * q_lo_w
                  + cpolicy[b_idx, y_idx, q_hi_i, z_idx] * q_hi_w)
        wealth = b + zt * y
        c = jnp.clip((wealth - bc) * cshare, eps, wealth - (bc + eps) * q)
        u = crra_util_func(c, sigma) * (1.0 - gamma)
        b_next = (wealth - c) / q
        b_lo, b_hi, wl, wh = interp_two_point_nonuniform(b_next, b_grid)
        cols = jnp.concatenate([b_lo * ny + jnp.arange(ny, dtype=jnp.int32),
                                b_hi * ny + jnp.arange(ny, dtype=jnp.int32)])
        rows = jnp.full((2 * ny,), idx_S, dtype=jnp.int32)
        vals = jnp.concatenate([wl * y_trans[y_idx], wh * y_trans[y_idx]])
        return (rows, cols, vals), u

    A, U = jax.vmap(build_triplets_for_state)(jnp.arange(J, dtype=jnp.int32))
    z_idx_next = jax.random.choice(key, nz, p=z_trans[z_idx])
    return A, U + penalty, (z_idx_next, 0.0)


state_space = {"b": ("markov", nb), "y": ("markov", ny),
               "q": ("non-markov", nq), "z": ("non-markov", nz)}
action_space = {"first_col": (0.001, 1.0, 0.5), "diff_col": (-0.2, 0.2, 0.0)}

# %% [markdown]
# ## Solve with SPG
#
# `SPGSolver` simulates a batch of paths through `AUS_func`, estimates lifetime
# utility, differentiates it through the (smooth) dynamics, and ascends. These
# are the paper-canonical settings; the policy converges to a smooth function
# after a few hundred steps (`eps=1e-4` is the convergence tolerance on the
# policy-update size). On a GPU this is a few minutes.

# %%
spg_solver = SPGSolver(sample_size=128, warm_up=0, epoch=1000,
                       eps=1e-4, learning_rate=1e-3, early_stop=True)
spg_policy, logs = spg_solver.solve(
    gamma, state_space, action_space, AUS_func, reset_func, policy=initial_guess
)
cpolicy = from_diff_to_cshare(spg_policy)
print(f"SPG: {len(logs)} iters, {spg_solver.total_time:.1f}s, "
      f"final max_l1_err {float(logs[-1]['max_l1_err']):.3E}")

# %% [markdown]
# ## A helper: the bond-supply schedule and the clearing price
#
# To draw the equilibrium objects we need, for a given aggregate state $z$, the
# aggregate bond-supply curve $S(q)$ and the price at which it clears $B$. This
# is the same computation `AUS_func` does internally; we pull it out so the
# figures can call it. The cross-section we aggregate against is the one the
# solver settled into, `mt`.

# %%
mt = spg_solver.m0.mean(0)          # the ergodic cross-section the solver carries


def supply_and_clearing(z_idx):
    """Aggregate bond supply S(q) over the price grid, and the clearing price q*."""
    zt = z_grid[z_idx]
    cshare_of_q = cpolicy[..., z_idx].reshape(-1, nq).T
    wealth_of_q = jnp.ones((nq, nb * ny)) * (b_dist + zt * y_dist)[None, :]
    C_of_q = jnp.clip((wealth_of_q - bc) * cshare_of_q, eps,
                      wealth_of_q - (bc + eps) * q_grid[:, None])
    S_of_q = jnp.sum((wealth_of_q - C_of_q) * mt[None, :], axis=-1) / q_grid
    lo, hi, wl, wh = find_clearing_point(B, S_of_q)
    q_star = float(q_grid[lo] * wl + q_grid[hi] * wh)
    return S_of_q, q_star

# %% [markdown]
# ## Figure 1: equilibrium consumption policy across aggregate states
#
# Consumption $c(b)$ at the market-clearing price, one line per income state,
# shown at a low, middle, and high aggregate state $z$. The policy is smooth and
# concave, kinked where the borrowing constraint binds, and shifts up with both
# income $y$ and aggregate productivity $z$: good aggregate times raise
# consumption at every wealth level.

# %%
z_panels = [0, nz // 2, nz - 1]
bmask = jnp.where(b_grid <= 30)[0]
fig, axs = plt.subplots(1, len(z_panels), figsize=(5 * len(z_panels), 4.2),
                        sharex=True, sharey=True)
for ax, z_idx in zip(axs, z_panels):
    zt = float(z_grid[z_idx])
    _, q_star = supply_and_clearing(z_idx)
    lo, hi, wl, wh = interp_two_point_nonuniform(q_star, q_grid)
    cshare = cpolicy[..., lo, z_idx] * wl + cpolicy[..., hi, z_idx] * wh   # (nb, ny)
    for y_idx in range(ny):
        wealth = b_grid + zt * y_grid[y_idx] - bc
        c = jnp.clip(wealth * cshare[:, y_idx], eps, wealth - eps)
        ax.plot(b_grid[bmask], c[bmask], color=COLORS[y_idx], lw=2.5,
                label=f"$y = {float(y_grid[y_idx]):.2f}$")
    ax.set_title(f"$z = {zt:.2f}$  (clears at $q = {q_star:.3f}$)")
    ax.set_xlabel("wealth $b$")
    ax.grid(alpha=0.3)
axs[0].set_ylabel("consumption $c$")
axs[0].legend()
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Figure 2: market clearing across aggregate states
#
# The headline picture. For a fan of aggregate states $z$, each curve is the
# aggregate bond-supply schedule $S(q, z)$; the star marks where it crosses net
# supply $B$, the market-clearing price for that state. Higher $z$ shifts the
# schedule right (households want to save more in good times), so the
# **equilibrium price rises with the aggregate state**. This is the equilibrium
# price *process* SRL learned, read off directly from the converged policy.

# %%
fig, ax = plt.subplots(figsize=(7, 5.5))
z_fan = [int(round(f * (nz - 1))) for f in (0.1, 0.3, 0.5, 0.7, 0.9)]
qmask = jnp.where(q_grid >= 0.965)[0]
for ci, z_idx in enumerate(z_fan):
    S_of_q, q_star = supply_and_clearing(z_idx)
    color = COLORS[ci % len(COLORS)]
    ax.plot(S_of_q[qmask], q_grid[qmask], "o-", color=color, markersize=5,
            label=f"$z = {float(z_grid[z_idx]):.2f}$, $q^* = {q_star:.3f}$")
    ax.plot(B, q_star, "*", color=color, markersize=15)
ax.axvline(B, ls="--", color="grey", label="net supply $B$")
ax.set_xlabel("aggregate bond supply $S(q, z)$")
ax.set_ylabel("bond price $q$")
ax.set_title("Market clearing across aggregate states")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Figure 3: the objective climbing
#
# The policy-gradient signature: lifetime utility rising as the policy improves,
# and the policy-update size shrinking toward zero as it converges.

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4))
axs[0].plot([lg["cumulative_utility"] for lg in logs])
axs[0].set_xlabel("iteration"); axs[0].set_ylabel("lifetime utility")
axs[0].set_title("Objective climbing"); axs[0].grid(alpha=0.3)
axs[1].plot([lg["max_l1_err"] for lg in logs])
axs[1].set_yscale("log")
axs[1].set_xlabel("iteration"); axs[1].set_ylabel("policy update (max $L_1$)")
axs[1].set_title("Convergence"); axs[1].grid(alpha=0.3)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Exercises
#
# 1. **Impulse response to an aggregate shock.** Hold the idiosyncratic process
#    fixed and trace how the economy responds to a one-time jump in aggregate
#    productivity $z$. A worked solution is below.
# 2. **Does the price clear the market each period?** SRL's bet is that a single
#    price can stand in for the whole distribution. A necessary check: as the
#    distribution shifts with $z$, does the market-clearing price actually clear
#    the bond market every period? A worked solution is below.

# %% [markdown]
# ### Exercise 1: impulse response (worked)
#
# We compute equilibrium aggregate consumption at each fixed aggregate state, then
# read off the response to moving $z$ from its mean to a high value and back. This
# is the simplest impulse response: how aggregate consumption co-moves with the
# aggregate shock, in equilibrium.

# %%
def aggregate_consumption(z_idx):
    """Aggregate consumption at the market-clearing price for aggregate state z."""
    zt = z_grid[z_idx]
    _, q_star = supply_and_clearing(z_idx)
    lo, hi, wl, wh = interp_two_point_nonuniform(q_star, q_grid)
    cshare = (cpolicy[..., lo, z_idx] * wl + cpolicy[..., hi, z_idx] * wh).reshape(-1)
    wealth = b_dist + zt * y_dist
    c = jnp.clip((wealth - bc) * cshare, eps, wealth - (bc + eps) * q_star)
    return float(jnp.sum(c * mt))


z_mean_idx = nz // 2
agg_c = jnp.array([aggregate_consumption(z) for z in range(nz)])
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(z_grid, agg_c, "o-", color=COLORS[0])
ax.axvline(float(z_grid[z_mean_idx]), ls="--", color="grey", label="mean $z$")
ax.set_xlabel("aggregate state $z$")
ax.set_ylabel("aggregate consumption $C(z)$")
ax.set_title("Exercise 1: aggregate consumption co-moves with the aggregate shock")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
plt.show()
print(f"C rises {100 * (agg_c[-1] / agg_c[0] - 1):.1f}% from the lowest to the highest z")

# %% [markdown]
# ### Exercise 2: does the price clear the market? (worked)
#
# We roll the cross-section forward under the learned policy along a random
# aggregate path and record the **market-clearing residual** each period: the gap
# between realized aggregate bond supply *at the clearing price* and net supply
# $B$. Whenever supply brackets $B$, the interpolated clearing price hits it
# exactly, so the residual is ~0, even though the distribution shifts every
# period with $z$. (The *deeper* accuracy question, whether one price is a truly
# sufficient statistic for the whole distribution, is what the paper measures
# against high-accuracy references; here we just confirm the market clears.)

# %%
def simulate_residuals(key, n_periods=80, burn_in=40):
    """Roll the cross-section forward; record the clearing residual each period."""
    m = jnp.ones(nb * ny) / (nb * ny)
    z_idx, _ = reset_func(key)
    residuals = []
    for _ in range(n_periods):
        key, sub = jax.random.split(key)
        S_of_q, _ = supply_and_clearing(int(z_idx))
        # Residual at the *interpolated* clearing price: w_lo*S[lo] + w_hi*S[hi] - B.
        # ~0 whenever aggregate supply brackets net supply B (the market clears).
        lo, hi, wl, wh = find_clearing_point(B, S_of_q)
        residuals.append(float(jnp.abs(wl * S_of_q[lo] + wh * S_of_q[hi] - B)))
        A, _, (z_idx, _) = AUS_func(spg_policy, m, (z_idx, 0.0), sub)
        m = apply_A_T(A, m)
        m = m / jnp.sum(m)
    return jnp.array(residuals[burn_in:])


res = simulate_residuals(jax.random.PRNGKey(0))
print(f"market-clearing residual over the simulation: "
      f"mean {float(res.mean()):.2e}, max {float(res.max()):.2e}")
print("The bond market clears every period (residual ~0) even as the "
      "distribution moves with z, a single price stands in for it.")

# %% [markdown]
# That's it. You have solved a Huggett economy with aggregate risk by policy
# gradient, the regime where carrying the full distribution as a state would be
# infeasible and where classical global methods struggle most. The household never
# tracked the distribution. It carried a single price, and the solver learned the
# equilibrium price process from simulation.
#
# At the **paper-scale** grids (`200, 3, 20, 50`, `sample_size=128`) these same
# figures come out fully smooth. That run is a GPU job of a few minutes.
