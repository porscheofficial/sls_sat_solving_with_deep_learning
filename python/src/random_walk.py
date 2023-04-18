import sys

sys.path.append("../../")

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.experimental.sparse import BCOO
from jraph._src import utils

from constraint_problems import SATProblem


def moser_walk(weights, problem, n_steps, seed, keep_trajectory=False):
    # noinspection PyShadowingNames
    @partial(jax.jit, static_argnames="prob")
    def step(state, prob: SATProblem):
        current, _, counter, rng_key = state
        _, rng_key = jax.random.split(rng_key)

        # identify violated constraint
        constraint_is_violated = violated_constraints(prob, current)
        j = np.argmax(constraint_is_violated)
        num_violations = np.sum(constraint_is_violated).reshape((-1,))
        e = len(prob.graph.edges)
        n, m, _ = prob.params
        adjacency_matrix = BCOO(
            (
                np.ones(e),
                np.column_stack((prob.graph.senders, prob.graph.receivers)),
            ),
            shape=(n, m),
        ).todense()

        randomness = jax.random.bernoulli(rng_key, weights)
        new = jnp.where(adjacency_matrix[:, j], randomness, current)
        return new, num_violations, counter + 1, rng_key

    def step_trajectory(state, prob: SATProblem):
        trajectory, energies, counter, rng_key = state
        inner_state = trajectory[counter, :], 0, counter, rng_key
        print(trajectory.shape)
        new, new_energy, updated_counter, new_rng_key = step(inner_state, prob)
        energies = energies.at[updated_counter].set(new_energy[0])
        trajectory = trajectory.at[updated_counter, :].set(new)
        return trajectory, energies, updated_counter, new_rng_key

    @partial(jax.jit, static_argnames="limit")
    def is_not_solution(state, limit):
        _, energy, counter, _ = state
        return (energy[-1] > 0) & (counter < limit)

    rng_key = jax.random.PRNGKey(seed)
    initial_assignment = jax.random.bernoulli(rng_key, weights).reshape(1, -1)
    constraint_is_violated = violated_constraints(problem, initial_assignment)
    energy = np.sum(constraint_is_violated).reshape((-1,))

    if keep_trajectory:
        step_func = step_trajectory
        initial_assignment = jnp.zeros((n_steps, weights.shape[0]), dtype=bool)
        initial_assignment.at[0].set(initial_assignment[0, :])
        energy = jnp.zeros((n_steps,))
        energy.at[0].set(energy[0])

    else:
        step_func = step

    output, energy, counter, _ = jax.lax.while_loop(
        cond_fun=partial(is_not_solution, limit=n_steps),
        body_fun=partial(step_func, prob=problem),
        init_val=(
            initial_assignment,
            energy,
            0,
            rng_key,
        ),
    )
    return output, energy, counter


@partial(jax.jit, static_argnames=("problem",))
def violated_constraints(problem: SATProblem, assignment):
    graph = problem.graph
    edge_is_violated = jnp.mod(graph.edges[:, 1] + assignment.T[graph.senders].T, 2)

    e = len(graph.edges)
    _, m, k = problem.params
    edge_mask_sp = BCOO(
        (np.ones(e), np.column_stack((np.arange(e), graph.receivers))), shape=(e, m)
    )

    violated_constraint_edges = edge_is_violated @ edge_mask_sp  # (,x) @ (x,m)  = (,m)
    constraint_is_violated = violated_constraint_edges == jnp.asarray(
        problem.clause_lengths
    )

    # constraint_is_violated = (
    #     jax.vmap(jnp.sum)(jnp.reshape(edge_is_violated, (m, k))) == k
    # )
    return constraint_is_violated


def number_of_violated_constraints_VCG(problem: SATProblem, assignment):
    return np.sum(violated_constraints(problem, assignment).astype(int), axis=0)


# @partial(jax.jit, static_argnames=("problem",))
def number_of_violated_constraints_LCG(problem: SATProblem, assignment):
    def one_hot(x, k, dtype=jnp.float32):
        """Create a one-hot encoding of x of size k."""
        return jnp.array(x[:, None] == jnp.arange(k), dtype)

    graph = problem.graph
    n, m, k = problem.params
    senders = graph.senders[:-n]
    receivers = graph.receivers[:-n]
    new_assignment = jnp.ravel(one_hot(assignment, 2))
    edge_is_satisfied = jnp.ravel(
        new_assignment[None].T[senders].T
    )  # + np.ones(len(senders)), 2)
    number_of_literals_satisfied = utils.segment_sum(
        data=edge_is_satisfied, segment_ids=receivers, num_segments=2 * n + m
    )[2 * n :]
    clause_is_unsat = jnp.where(number_of_literals_satisfied > 0, 0, 1)
    print(clause_is_unsat)
    return jnp.sum(clause_is_unsat)


"""
# @partial(jax.jit, static_argnames=("problem",))
def violated_constraints_VCG(problem: SATProblem, assignment):
    def one_hot(x, k, dtype=jnp.float32):
        '''Create a one-hot encoding of x of size k.'''
        return jnp.array(x[:, None] == jnp.arange(k), dtype)
        #return jnp.array(x[:, None] == jnp.arange(k), dtype)
    graph = problem.graph
    e = len(graph.edges)
    n, m, k = problem.params
    senders = graph.senders
    edges = graph.edges
    receivers = graph.receivers
    print("edges")
    print(edges)
    new_assignment = one_hot(assignment.T,2)
    edge_is_satisfied = new_assignment[senders] * edges[senders] # + np.ones(len(senders)), 2)
    print("sat",edge_is_satisfied.shape)
    edge_is_satisfied = jnp.sum(edge_is_satisfied,axis=1)
    print("sat sum", edge_is_satisfied.shape)
    number_of_literals_satisfied = utils.segment_sum(data=edge_is_satisfied, segment_ids=receivers, num_segments=n + m)[n:]
    print(number_of_literals_satisfied)
    clause_is_unsat = jnp.where(number_of_literals_satisfied > 0, 0, 1)
    print("clause unsat", clause_is_unsat)
    print(clause_is_unsat.shape) 
    return jnp.sum(clause_is_unsat)
"""
# %%
