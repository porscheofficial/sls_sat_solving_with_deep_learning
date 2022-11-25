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
            (np.ones(e), np.column_stack((prob.graph.senders, prob.graph.receivers))),
            shape=(n, m),
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
        initial_assignment = jnp.zeros((n_steps, weights.shape[0]))
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


# def moser_walk_sampler(weights, problem, n_steps, seed):
#     """
#     This is a modified moser walker that will run [n_samples] iterations of the random walk even
#     if it has found a solution already.
#     """
#
#     def step(i, state, prob: SATProblem):
#         assignments, energies = state
#         # graph = prob.graph
#
#         # identify violated constraint
#         current = assignments[i, :]
#         constraint_is_violated = violated_constraints(prob, current)
#         j = np.argmax(constraint_is_violated)  # , size=1)[0]
#         num_violations = np.sum(constraint_is_violated)
#         # evalute violations
#         energies = energies.at[i].set(num_violations)
#         # resample the first violated constraint (or the first constraint if none are violated)
#         edge_mask, clause_lengths, constraint_mask = prob.constraint_utils
#
#         # cumulative_lengths = np.cumsum(np.array(clause_lengths), dtype=np.int32)
#         # jax.debug.print("{0}", cumulative_lengths)
#         # init_index = jax.lax.dynamic_index_in_dim(cumulative_lengths, j)
#         # support_size = jax.lax.dynamic_index_in_dim(np.array(clause_lengths), j)
#         # support = jax.lax.dynamic_slice(graph.senders, init_index, (support_size,))
#         # jax.lax.dynamic_index_in_dim(jnp.array(clause_lengths), j)
#         randomness = assignments[i + 1, :]
#         new = jnp.where(jnp.asarray(constraint_mask)[:, j], randomness, current)
#         assignments = assignments.at[i + 1, :].set(new)
#         return assignments, energies
#
#     rng_key = jax.random.PRNGKey(seed)
#     random_assignments = jax.random.bernoulli(
#         rng_key, weights, (n_steps, weights.shape[0])
#     )
#     init_energies = np.zeros(n_steps, dtype=np.int32)
#     trajectory, energy = jax.lax.fori_loop(
#         0, n_steps, partial(step, prob=problem), (random_assignments, init_energies),
#     )
#
#     return trajectory, energy


@partial(jax.jit, static_argnames=("problem",))
def violated_constraints(problem: SATProblem, assignment):
    graph = problem.graph
    edge_is_violated = jnp.mod(graph.edges[:, 1] + assignment[graph.senders].T, 2)

    # we can generate a sparse matrix here. This should be compiled once initially.
    e = len(graph.edges)
    _, m, k = problem.params
    edge_mask_sp = BCOO(
        (np.ones(e), np.column_stack((np.arange(e), graph.receivers))), shape=(e, m)
    )
    # we changed this to deal with general constraint problems:
    # we hand down a list of constraint lengths.
    # we introduce an "edge mask", a weighted matrix of size (number of edges) x m. We then multiply this
    # with edge_is_violated and finally check whether the result lies below the number of constraints
    # edge_mask, _, _ = problem.constraint_utils
    violated_constraint_edges = edge_is_violated @ edge_mask_sp  # (x,) @ (x,m)  = (m,)

    # teh edge mask is weighted, a constraint is violated iff the violated edge weights sum to 1.
    constraint_is_violated = violated_constraint_edges == problem.clause_lengths

    # constraint_is_violated = (
    #     jax.vmap(jnp.sum)(jnp.reshape(edge_is_violated, (m, k))) == k
    # )
    return constraint_is_violated

# %%
