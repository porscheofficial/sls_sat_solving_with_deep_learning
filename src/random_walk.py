import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from constraint_problems import violated_constraints, SATProblem


def moser_walk_sampler(weights, problem, n_steps):
    """
    This is a modified moser walker that will run [n_samples] iterations of the random walk even
    if it has found a solution already.
    """
    n, m, k = problem.params

    def step(i, state, prob: SATProblem):
        assignments, energies = state
        # graph = prob.graph

        # identify violated constraint
        current = assignments[i, :]
        constraint_is_violated = violated_constraints(prob, current)
        j = np.argmax(constraint_is_violated)  # , size=1)[0]
        num_violations = np.sum(constraint_is_violated)
        # evalute violations
        energies = energies.at[i].set(num_violations)
        # resample the first violated constraint (or the first constraint if none are violated)
        edge_mask, clause_lengths, constraint_mask = prob.constraint_utils

        # cumulative_lengths = np.cumsum(np.array(clause_lengths), dtype=np.int32)
        # jax.debug.print("{0}", cumulative_lengths)
        # init_index = jax.lax.dynamic_index_in_dim(cumulative_lengths, j)
        # support_size = jax.lax.dynamic_index_in_dim(np.array(clause_lengths), j)
        # support = jax.lax.dynamic_slice(graph.senders, init_index, (support_size,))
        jax.lax.dynamic_index_in_dim(np.array(clause_lengths), j)
        randomness = assignments[i + 1, :]
        new = jnp.where(jnp.asarray(constraint_mask)[:, j], randomness, current)
        assignments = assignments.at[i + 1, :].set(new)
        return assignments, energies

    rng_key = jax.random.PRNGKey(42)
    random_assignments = jax.random.bernoulli(rng_key, weights, (n_steps, n))
    init_energies = np.zeros(n_steps, dtype=np.int32)
    trajectory, energy = jax.lax.fori_loop(
        0, n_steps, partial(step, prob=problem), (random_assignments, init_energies),
    )

    return trajectory, energy

#%%
