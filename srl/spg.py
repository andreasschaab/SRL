"""Stochastic Policy Gradient (SPG): solving HA models by differentiable simulation.

This is the core SRL solver. If you know value-function iteration (see
``vfi.py``), here is the contrast: VFI computes the value function and *reads off*
the optimal policy; SPG never forms a value function at all. Instead it
parameterises the policy directly as a tensor over the state grid and improves it
by gradient ascent on simulated lifetime utility, the same "just follow the
gradient of the objective" idea behind modern policy-gradient RL, but with the
gradient computed exactly by automatic differentiation rather than estimated from
noisy rollouts.

The algorithm, in one paragraph
-------------------------------
Hold a candidate policy ``pi`` (a tensor of, e.g., consumption shares over the
discretised state space). Simulate the economy forward for ``T`` periods: each
period the model hands back a sparse transition matrix ``A`` (how the
distribution of agents moves), a utility vector ``U`` (flow payoff at each state),
and the next exogenous state. Accumulate discounted utility along the path to get
a scalar objective ``J(pi)``. Because every step (``A``, ``U``, the
distribution push-forward) is written in JAX, ``J(pi)`` is differentiable in
``pi``: one ``jax.value_and_grad`` call gives the exact policy gradient
``dJ/dpi``. Take an Adam ascent step, clip the policy back into its admissible
range, and repeat until the policy stops moving.

The objective and the two distributions
---------------------------------------
Each simulated step the model returns ``A, U, s'`` from the user's ``AUS_func``.
We carry two distributions over the ``J`` endogenous states:

  * ``m`` is the **population** distribution. It is detached (``stop_gradient``)
    before being passed to ``AUS_func``, so general-equilibrium objects that
    depend on the cross-sectional distribution (e.g. a market-clearing price) see
    the *actual* population without the policy gradient flowing through that
    channel. Both ``m`` and ``d`` are pushed forward by the same ``A`` each step.
  * ``d`` is the **differentiable** distribution used to weight per-state utility
    ``U`` into the scalar objective ``sum_t gamma^t * (U_t . d_t)``. The gradient
    *does* flow through ``d`` (and through ``U`` and ``A``), and that is what
    makes ``J(pi)`` differentiable.

We average the objective over ``sample_size`` independent simulated paths (each
with its own random exogenous shocks) to reduce variance, and *minimise* its
negative (optimisers minimise by convention).

The user supplies the economics through two callbacks:
  * ``AUS_func(policy, m, s, key) -> (A, U, s')``: the model's one-step map.
  * ``reset_func(key) -> s0``: draws an initial exogenous state.
``A`` is a sparse triplet ``(rows, cols, vals)``; see ``utils.safe_linalg``.
"""
import time
from contextlib import contextmanager
from copy import deepcopy
from functools import partial

import jax
import jax.numpy as jnp
import optax
from tqdm import tqdm

from srl.utils.safe_linalg import apply_A_T, generate_uniform_m0


class SPGSolver:
    """Stochastic Policy Gradient solver for heterogeneous-agent models.

    Single-device (CPU or one GPU). Configure once, then call :meth:`solve`.

    Args:
        eps:            convergence bound on the max per-iteration policy change.
        sample_size:    number of independent simulated paths averaged per update.
        epoch:          maximum number of gradient-ascent updates.
        warm_up:        number of initial updates that (a) hold the learning rate
                        flat and (b) draw a fresh random starting distribution each
                        time, to shake the policy loose of its initial guess.
        noise_m0:       during warm-up, randomise the starting distribution per path
                        (vs. starting every path uniform).
        learning_rate:  initial Adam step size.
        lr_decay:       final-to-initial learning-rate ratio (exponential schedule).
        max_grad_norm:  global-norm gradient-clipping threshold.
        trunc_eps:      truncate each simulation at horizon T where gamma**T < trunc_eps
                        (the tail beyond it is negligibly discounted).
        early_stop:     stop once the convergence bound is met.
        seed:           PRNG seed.
        verbose:        print policy shapes and a per-iteration progress bar.
    """

    def __init__(
        self,
        eps: float = 0.001,
        sample_size: int = 512,
        epoch: int = 1000,
        warm_up: int = 50,
        noise_m0: bool = True,
        learning_rate: float = 0.01,
        lr_decay: float = 0.1,
        max_grad_norm: float = 10.0,
        trunc_eps: float = 0.001,
        early_stop: bool = True,
        seed: int = 0,
        verbose: bool = True,
    ) -> None:
        self.eps = eps
        self.sample_size = sample_size
        self.epoch = epoch
        self.warm_up = warm_up
        self.noise_m0 = noise_m0
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.max_grad_norm = max_grad_norm
        self.trunc_eps = trunc_eps
        self.early_stop = early_stop
        self.seed = seed
        self.verbose = verbose

    # NOTE on ``@partial(jax.jit, static_argnums=(0,))`` below: ``self`` is the
    # static argument. After :meth:`solve` has set the configuration and the
    # ``AUS_func`` / ``reset_func`` callbacks, ``self`` is effectively a frozen
    # bundle of constants, so JAX can treat it as compile-time-static and compile
    # the simulation once.

    # -- the objective: expected discounted utility from forward simulation ----

    @partial(jax.jit, static_argnums=(0,))
    def _neg_simulated_utility(self, policy, key, m0):
        """Return (-mean discounted lifetime utility, final population dist).

        Averages over ``sample_size`` independent paths. Negated because the
        optimiser minimises.
        """
        def simulate_one_path(key, m0):
            def step(carry, _):
                m, d, s, discount, key = carry
                # The environment sees the population distribution, detached so the
                # gradient does not flow through the GE feedback channel.
                m_detached = jax.lax.stop_gradient(m)
                m_detached = m_detached / jnp.sum(m_detached, keepdims=True)
                key, subkey = jax.random.split(key)
                A, U, s = self.env_AUS_fn(policy, m_detached, s, subkey)
                # Flow contribution to the objective: utility weighted by the
                # differentiable distribution d, discounted to the present.
                flow_utility = jnp.sum(U * d) * discount
                discount = discount * self.gamma
                # Push both distributions forward one period: x_{t+1} = A^T x_t.
                d = apply_A_T(A, d)
                m = apply_A_T(A, m)
                return (m, d, s, discount, key), flow_utility

            key, subkey = jax.random.split(key)
            s0 = self.env_reset_fn(subkey)
            d0 = jnp.ones(self.J) / self.J  # differentiable dist starts uniform
            init = (m0, d0, s0, 1.0, key)
            (m_final, _d, _s, _disc, _key), flow_utilities = jax.lax.scan(
                step, init, length=self.trunc_len
            )
            return flow_utilities.sum(), m_final  # population dist after T steps

        keys = jax.random.split(key, self.sample_size)
        path_utilities, m_final = jax.vmap(simulate_one_path, in_axes=(0, 0))(keys, m0)
        return -path_utilities.mean(), m_final

    @partial(jax.jit, static_argnums=(0,))
    def _gradient_step(self, policy, opt_state, key, m0):
        """One Adam ascent step on the simulated-utility objective.

        Returns the updated policy, optimiser state, the (positive) objective
        value, and the final population distribution from the simulation.
        """
        (neg_utility, m_final), grads = jax.value_and_grad(
            self._neg_simulated_utility, has_aux=True
        )(policy, key, m0)
        updates, opt_state = self.optimizer.update(grads, opt_state)
        policy = optax.apply_updates(policy, updates)
        return policy, opt_state, -neg_utility, m_final

    @partial(jax.jit, static_argnums=(0,))
    def _warmup_step(self, policy, opt_state, key):
        """A warm-up update: draw a fresh starting distribution, then ascend.

        Randomising the initial distribution per path (when ``noise_m0``) keeps the
        early policy from overfitting one particular starting cross-section.
        """
        key_m0, key_sim = jax.random.split(key)
        if self.noise_m0:
            per_path_keys = jax.random.split(key_m0, self.sample_size)
            m0 = jax.vmap(generate_uniform_m0, in_axes=(0, None))(per_path_keys, self.J)
        else:
            m0 = jnp.ones((self.sample_size, self.J)) / self.J
        policy, opt_state, utility, _ = self._gradient_step(policy, opt_state, key_sim, m0)
        return policy, opt_state, utility

    def _ergodic_step(self):
        """A post-warm-up update: reuse the population distribution reached so far.

        Carrying ``m0`` forward across updates lets the simulation start from a
        near-ergodic distribution, so each step refines the policy on the states
        the economy actually visits.
        """
        key, self.key = jax.random.split(self.key)
        policy, self.opt_state, utility, self.m0 = self._gradient_step(
            self.policy, self.opt_state, key, self.m0
        )
        self.m0 = self.m0 / jnp.sum(self.m0, -1, keepdims=True)
        return policy, utility.item()

    def solve(self, gamma, state_space, action_space, env_AUS_func, env_reset_func, policy=None):
        """Train the policy by stochastic policy-gradient ascent.

        Args:
            gamma:          discount factor (float in (0, 1)).
            state_space:    dict ``{var: ("markov"|"non-markov", n_grid)}``. The
                            "markov" variables span the endogenous distribution
                            (their product is ``J``); "non-markov" variables are
                            exogenous states carried in ``s``.
            action_space:   dict ``{action: (lower, upper, init_value)}``.
            env_AUS_func:   ``(policy, m, s, key) -> (A, U, s')``, the model map.
            env_reset_func: ``(key) -> s0``, draws an initial exogenous state.
            policy:         optional initial policy tensors (else filled with each
                            action's ``init_value``).

        Returns:
            ``(policy, logs)``: the trained policy and the per-iteration trace of
            cumulative utility and max policy change.
        """
        self._validate(gamma, state_space, action_space, env_AUS_func, env_reset_func)
        self.gamma = gamma
        # Truncate each simulation where the discount makes the tail negligible.
        self.trunc_len = jnp.ceil(jnp.log(self.trunc_eps) / jnp.log(gamma)).astype(int)
        if self.verbose:
            print(f"trunc_T={self.trunc_len}")

        self.total_time = 0
        with self._timer():
            self._init_policy(state_space, action_space, policy)
            lr_scheduler = optax.exponential_decay(
                self.learning_rate, self.epoch, self.lr_decay, self.warm_up
            )
            # Clip the gradient's global norm, then Adam with the decay schedule.
            self.optimizer = optax.chain(
                optax.clip_by_global_norm(self.max_grad_norm),
                optax.adam(lr_scheduler, b1=0.9, b2=0.999),
            )
            self.opt_state = self.optimizer.init(self.policy)

        self.env_AUS_fn = env_AUS_func
        self.env_reset_fn = env_reset_func
        self.key = jax.random.PRNGKey(self.seed)
        self.m0 = jnp.ones((self.sample_size, self.J)) / self.J

        logs = []
        converged = False
        with tqdm(range(self.epoch)) as pbar:
            for i in pbar:
                with self._timer():
                    if i < self.warm_up:
                        key, self.key = jax.random.split(self.key)
                        policy_new, self.opt_state, utility = self._warmup_step(
                            self.policy, self.opt_state, key
                        )
                    else:
                        policy_new, utility = self._ergodic_step()
                    # Project each action back into its admissible [lower, upper].
                    policy_new = {
                        k: jnp.clip(v, *self.policy_range[k]) for k, v in policy_new.items()
                    }
                    # Convergence metric: a high quantile of the per-element policy
                    # change (robust to a few boundary elements that keep moving).
                    change = jax.tree_util.tree_map(
                        lambda old, new: jnp.quantile(jnp.abs(old - new), 0.999),
                        self.policy, policy_new,
                    )
                    self.policy = policy_new
                    max_change = max(c.item() for c in change.values())
                    # Converged once the policy barely moves: max_change <= ~eps.
                    # (round(x/eps, 1) <= 1 is "x is at most about one eps", with a
                    # little slack from rounding.)
                    if self.early_stop and i >= self.warm_up and round(max_change / self.eps, 1) <= 1:
                        converged = True
                pbar.set_postfix({
                    "max_l1_err": f"{max_change:.3E}",
                    "time(sec)": f"{self.total_time:.3f}",
                    "cum(EU)": f"{utility:.4f}",
                })
                logs.append({"cumulative_utility": utility, "max_l1_err": max_change})
                if converged:
                    break
        return self.policy, logs

    # -- setup helpers --------------------------------------------------------

    def _init_policy(self, state_space, action_space, policy=None):
        """Allocate the policy tensor(s) and record the endogenous-state count J."""
        self.shapes = [n for _, n in state_space.values()]
        if policy is not None:
            self.policy = deepcopy(policy)
        else:
            self.policy = {
                name: jnp.ones(self.shapes) * init_v
                for name, (_, _, init_v) in action_space.items()
            }
        self.policy_range = {name: (lo, hi) for name, (lo, hi, _) in action_space.items()}
        # J = number of endogenous (markov) states the distribution lives on.
        self.J = 1
        for kind, n in state_space.values():
            if kind == "markov":
                self.J *= n
        if self.verbose:
            print([f"{k}.shape = {v.shape}" for k, v in self.policy.items()], f"J = {self.J}")

    @staticmethod
    def _validate(gamma, state_space, action_space, env_AUS_func, env_reset_func):
        if not (isinstance(gamma, float) and 0 < gamma < 1):
            raise ValueError("gamma must be a float in (0, 1).")
        for kind, n in state_space.values():
            if kind not in ("markov", "non-markov"):
                raise ValueError("state_space[var][0] must be 'markov' or 'non-markov'.")
            if not isinstance(n, int) or n <= 0:
                raise ValueError("state_space[var][1] must be a positive integer.")
        for bounds in action_space.values():
            if not all(isinstance(x, float) for x in bounds):
                raise ValueError("action_space values must be (float, float, float).")
        if not callable(env_AUS_func) or not callable(env_reset_func):
            raise TypeError("env_AUS_func and env_reset_func must be callables.")

    @contextmanager
    def _timer(self):
        start = time.time()
        try:
            yield
        finally:
            self.total_time += time.time() - start
