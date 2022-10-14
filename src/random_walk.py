import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from constraint_problems import violated_constraints


def moser_walk_sampler(weights, problem, n_steps):
    """
    This is a modified moser walker that will run [n_samples] iterations of the random walk even
    if it has found a solution already.
    """
    params = list(problem.meta.values())
    n, m, k = params

    # we assume that the instance is k-CNF, so all clauses have the same length

    def step(i, state, prob):
        assignments, energies = state
        graph = prob.graph

        # identify violated constraint
        current = assignments[i, :]
        constraint_is_violated = violated_constraints(prob, current)
        j = jnp.where(constraint_is_violated, size=1)[0]
        num_violations = jnp.sum(constraint_is_violated)
        # evalute violations
        energies = energies.at[i].set(num_violations)

        # resample the first violated constraint (or the first constraint if none are violated)
        support = jax.lax.dynamic_slice(graph.senders, j * k, (k,))
        current = assignments[i, :]
        randomness = assignments[i + 1, :]
        new = current.at[support].set(randomness[support])
        assignments = assignments.at[i + 1, :].set(new)
        return assignments, energies

    rng_key = jax.random.PRNGKey(42)
    random_assignments = jax.random.bernoulli(rng_key, weights, (n_steps, n))
    init_energies = jnp.zeros(n_steps, dtype=np.int32)
    trajectory, energy = jax.lax.fori_loop(
        0,
        n_steps,
        partial(step, problem=problem),
        (random_assignments, init_energies),
    )

    return trajectory, energy
