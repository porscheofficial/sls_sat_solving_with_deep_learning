import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from jax.experimental.sparse import BCOO
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
        n, m, k = prob.params
        adjacency_matrix = BCOO(
            (np.ones(e, dtype=bool), np.column_stack((prob.graph.senders, prob.graph.receivers))),
            shape=(n, m)
        ).todense()

        randomness = jax.random.bernoulli(rng_key, weights)
        # noinspection PyShadowingBuiltins
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
        energy = jnp.zeros((n_steps, ))
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
        )
    )
    return output, energy, counter


@partial(jax.jit, static_argnames=("problem",))
#@jax.jit
def violated_constraints(problem: SATProblem, assignment):
    graph = problem.graph
    edge_is_violated = jnp.mod(graph.edges[:, 1] + assignment[graph.senders].T, 2)

    e = len(graph.edges)
    _, m, k = problem.params
    edge_mask_sp = BCOO(
        (np.ones(e), np.column_stack((np.arange(e), graph.receivers))), shape=(e, m)
    )

    violated_constraint_edges = edge_is_violated @ edge_mask_sp  # (,x) @ (x,m)  = (,m)
    constraint_is_violated = violated_constraint_edges == jnp.asarray(problem.clause_lengths)

    # constraint_is_violated = (
    #     jax.vmap(jnp.sum)(jnp.reshape(edge_is_violated, (m, k))) == k
    # )
    return constraint_is_violated

@partial(jax.jit, static_argnames=("problem",))
#@jax.jit
def number_of_violated_constraints(problem: SATProblem, assignment):
    graph = problem.graph
    edge_is_violated = jnp.mod(graph.edges[:, 1] + assignment[graph.senders].T, 2)

    e = len(graph.edges)
    _, m, k = problem.params
    edge_mask_sp = BCOO(
        (np.ones(e), np.column_stack((np.arange(e), graph.receivers))), shape=(e, m)
    )

    violated_constraint_edges = edge_is_violated @ edge_mask_sp  # (,x) @ (x,m)  = (,m)
    constraint_is_violated = violated_constraint_edges == jnp.asarray(problem.clause_lengths)

    # constraint_is_violated = (
    #     jax.vmap(jnp.sum)(jnp.reshape(edge_is_violated, (m, k))) == k
    # )
    return np.sum(constraint_is_violated.astype(int),axis=0)

# %%
