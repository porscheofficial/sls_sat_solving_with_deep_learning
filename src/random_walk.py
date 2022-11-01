import jax
import jax.numpy as jnp
import numpy as np
from functools import partial

from constraint_problems import violated_constraints, SATProblem


def moser_walk(weights, problem, n_steps, seed):

    @partial(jax.jit, static_argnames=("prob"))
    def step(state, prob: SATProblem):
        current, _, counter, rng_key = state
        _, rng_key = jax.random.split(rng_key)

        # identify violated constraint
        constraint_is_violated = violated_constraints(prob, current)
        j = np.argmax(constraint_is_violated)  # , size=1)[0]
        num_violations = np.sum(constraint_is_violated)
        # resample the first violated constraint (or the first constraint if none are violated)
        _, _, constraint_mask = prob.constraint_utils

        randomness = jax.random.bernoulli(rng_key, weights)
        next = jnp.where(constraint_mask[:, j], randomness, current)
        return next, num_violations, counter + 1, rng_key

    @partial(jax.jit, static_argnames=("limit"))
    def is_solution(state, limit):
        _, energy, counter, _ = state
        return (energy > 0) & (counter < limit)


    rng_key = jax.random.PRNGKey(seed)
    random_assignment = jax.random.bernoulli(rng_key, weights)
    constraint_is_violated = violated_constraints(problem, random_assignment)
    num_violations = np.sum(constraint_is_violated)
    trajectory, energy, _, _ = jax.lax.while_loop(
        cond_fun=partial(is_solution, limit=n_steps),
        body_fun=partial(step, prob=problem),
        init_val=(random_assignment, num_violations, np.int64(1), rng_key),
    )

    return trajectory, energy


def moser_walk_sampler(weights, problem, n_steps, seed):
    """
    This is a modified moser walker that will run [n_samples] iterations of the random walk even
    if it has found a solution already.
    """

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
        # jax.lax.dynamic_index_in_dim(jnp.array(clause_lengths), j)
        randomness = assignments[i + 1, :]
        new = jnp.where(jnp.asarray(constraint_mask)[:, j], randomness, current)
        assignments = assignments.at[i + 1, :].set(new)
        return assignments, energies

    rng_key = jax.random.PRNGKey(seed)
    random_assignments = jax.random.bernoulli(
        rng_key, weights, (n_steps, weights.shape[0])
    )
    init_energies = np.zeros(n_steps, dtype=np.int32)
    trajectory, energy = jax.lax.fori_loop(
        0, n_steps, partial(step, prob=problem), (random_assignments, init_energies),
    )

    return trajectory, energy


#%%
