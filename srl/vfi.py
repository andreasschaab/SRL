"""Value Function Iteration (VFI): the textbook dynamic-programming baseline.

VFI is the method most macroeconomists already know. We keep it in SRL as an
independent yardstick: when VFI and the policy-gradient solver (`SPGSolver`)
solve the *same* model and agree, that is strong evidence the SRL solution is
correct. The PE-Huggett tutorial uses exactly this cross-check.

The idea, in one paragraph
--------------------------
A stationary dynamic program is summarised by its value function ``V(s)``: the
expected discounted lifetime payoff of being in state ``s`` and behaving
optimally thereafter. ``V`` is the fixed point of the **Bellman operator**

    V(s) = max_a  [ reward(s, a) + gamma * E[ V(s') | s, a ] ].

VFI just applies that operator over and over: start from ``V = 0``, sweep every
state taking the best action, and repeat until ``V`` stops moving. The operator
is a contraction (modulus ``gamma``), so the iteration converges to the unique
fixed point from any starting guess.

The action space is a finite grid of ``n_action`` choices, so the ``max`` is a
plain maximisation over that grid at each state. The user supplies one function,
``transition_func(state_indices, action_idx, V) -> (E[V'], reward)``, that
encodes the model's economics (budget constraint, law of motion, expectations);
this solver supplies only the generic iteration around it.
"""
import math
import time
from contextlib import contextmanager
from functools import partial

import jax
import jax.numpy as jnp
from tqdm import tqdm


class VFISolver:
    """Value Function Iteration over a discrete action grid.

    Args:
        eps:      convergence tolerance on ``max |V_new - V_old|``.
        max_iter: cap on the number of Bellman sweeps.
        verbose:  print the shape of ``V`` and a per-sweep progress bar.
    """

    def __init__(self, eps: float = 1e-6, max_iter: int = 1000, verbose: bool = True) -> None:
        self.eps = float(eps)
        self.max_iter = int(max_iter)
        self.verbose = verbose

    # -- the Bellman operator -------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _bellman(self, V):
        """Apply the Bellman operator once: return the greedy policy and updated V.

        The state space is an arbitrary-dimensional grid (e.g. ``(nb, ny, nq, nz)``
        for PE Huggett). Rather than nest one ``vmap`` per dimension, we flatten
        the grid to a single axis of ``prod(shape)`` states, ``vmap`` over it, and
        reshape back: same computation, far less code, any number of dimensions.
        """
        shape = self.shape
        n_states = math.prod(shape)
        n_action = self.n_action
        gamma = self.gamma
        transition_func = self.transition_func

        def best_at(flat_idx):
            # Recover the multi-dimensional state index (b_idx, y_idx, ...).
            state_indices = jnp.unravel_index(flat_idx, shape)

            # Value of every candidate action: reward now + discounted continuation.
            def action_value(a_idx):
                EV_next, reward = transition_func(state_indices, a_idx, V)
                return reward + gamma * EV_next

            values = jax.vmap(action_value)(jnp.arange(n_action))
            return jnp.argmax(values), jnp.max(values)  # greedy action, V(s)

        greedy_action, V_new = jax.vmap(best_at)(jnp.arange(n_states))
        return greedy_action.reshape(shape), V_new.reshape(shape)

    def solve(self, gamma, state_space, action_space, transition_func):
        """Iterate the Bellman operator to its fixed point.

        Args:
            gamma:           discount factor in (0, 1).
            state_space:     dict ``{var_name: n_grid}``, the state-grid shape.
            action_space:    int, number of points on the discrete action grid.
            transition_func: ``(state_indices, action_idx, V) -> (E[V'], reward)``.
                             Encodes the model: ``state_indices`` is the tuple of
                             per-dimension grid indices, ``action_idx`` indexes the
                             action grid, and ``V`` is the current value array.

        Returns:
            ``(V, greedy_policy, logs)``: the converged value array, the greedy
            action index at every state, and the per-sweep error trace.
        """
        self.gamma = float(gamma)
        self.shape = tuple(state_space.values())
        self.n_action = action_space
        self.transition_func = transition_func

        self.total_time = 0.0
        V = jnp.zeros(self.shape)
        if self.verbose:
            print(f"V.shape={V.shape}")

        # Iterate V <- Bellman(V) until it stops moving (or we hit max_iter).
        logs = []
        with tqdm(range(self.max_iter)) as pbar:
            for _ in pbar:
                with self._timer():
                    _, V_new = self._bellman(V)
                    err = jnp.max(jnp.abs(V_new - V))
                    V = V_new
                logs.append({"max_l1_err": err, "time(sec)": self.total_time})
                pbar.set_postfix({"max_l1_err": f"{err:.3E}", "time(sec)": f"{self.total_time:.3f}"})
                if err < self.eps:
                    break

        # One final sweep to read off the greedy policy at the converged V.
        greedy_policy, _ = self._bellman(V)
        return V, greedy_policy, logs

    @contextmanager
    def _timer(self):
        start = time.time()
        try:
            yield
        finally:
            self.total_time += time.time() - start
