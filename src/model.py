import jraph
import jax
import haiku as hk
import optax
from functools import partial
import jax.numpy as jnp
import numpy as np

#from random_walk import moser_walk_sampler


def network_definition(
        graph: jraph.GraphsTuple, num_message_passing_steps: int = 5
) -> jraph.ArrayTree:
    """Defines a graph neural network.
    Args:
      graph: Graphstuple the network processes.
      num_message_passing_steps: number of message passing steps.
    Returns:
      Decoded nodes.
    """
    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=jax.vmap(hk.Linear(output_size=16)),
        embed_node_fn=jax.vmap(hk.Linear(output_size=16)),
    )
    graph = embedding(graph)

    def mlp(dims):
        net = []
        for d in dims:
            net.extend([hk.Linear(d), jax.nn.relu])
        return hk.Sequential(net)

    @jax.vmap
    @jraph.concatenated_args
    def update_fn(features):
        net = mlp([20, 20, 20])
        return net(features)

    for _ in range(num_message_passing_steps):
        gn = jraph.InteractionNetwork(
            update_edge_fn=update_fn,
            update_node_fn=update_fn,
            include_sent_messages_in_node_update=True,
        )
        graph = gn(graph)

    return hk.Linear(2)(graph.nodes)

'''
def train_model(
        sample_steps: int,
        field_strength: float,
        train_dataset,
        test_dataset,
        num_steps=None,
):
    if not num_steps:
        num_steps = len(train_dataset)

    seed = jax.random.PRNGKey(42)
    random_instances = jax.random.randint(
        seed, shape=(num_steps,), minval=0, maxval=len(train_dataset)
    )

    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(seed, train_dataset[random_instances[0]].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def supervised_prediction_loss(ps, problem):
        decoded_nodes = network.apply(ps, problem.graph)
        # We interpret the decoded nodes as a pair of logits for each node.
        log_prob = jax.nn.log_softmax(decoded_nodes) * problem.labels
        return -jnp.sum(log_prob * problem.mask[:, None]) / jnp.sum(problem.mask)

    # @partial(jax.jit, static_argnames=("s", "f"))
    # @partial(jax.jit, static_argnums=(0, 1))

    def run_sampler(s: int, sd: int, ps, problem):
        logits = network.apply(ps, problem.graph)

        # here we feed the model the second column of the output, since that is the probability that a "1" assignment
        # for each variable should be drawn
        probs = jax.nn.softmax(logits)[:, 1]
        trajectoriy, energies = moser_walk_sampler(probs, problem, s, sd)
        return trajectoriy, energies, logits

    def unsupervised_prediction_loss(s: int, f: float, sd: int, ps, problem):

        trajectory, energies, logits = run_sampler(s, sd, ps, problem)
        one_hot_encoded_trajectories = jnp.eye(2)[trajectory.astype(dtype=np.int32)]
        trajectory_log_probs = jnp.tensordot(
            one_hot_encoded_trajectories,
            jax.nn.log_softmax(logits) * problem.mask[:, None],
            2,
        )
        # calculating the weights numerically stably.
        # weighted_energies =
        # mean_energy = jnp.mean(weighted_energies)
        weights = jax.nn.softmax(- f * energies)
        weighted_log_probs = jnp.dot(trajectory_log_probs, weights)
        return -weighted_log_probs / jnp.sum(problem.mask)

    # Really interesting: I seem to get worse training results when I jit.

    # @partial(jax.jit, static_argnums=(0, 1))
    def update(s, f, sd, ps, opt, problem):
        loss = partial(unsupervised_prediction_loss, s, f, sd)
        g = jax.grad(loss)(ps, problem)
        updates, opt = opt_update(g, opt)
        return optax.apply_updates(params, updates), opt

    for i in range(num_steps):
        # in case num_steps is larger than the training set, we just iterate through the training set.

        instance = train_dataset[random_instances[i]]
        seed = i
        # pad graph for jitting
        # max_n_node = 5000
        # max_n_edge = 10000
        # instance.graph =

        params, opt_state = update(
            sample_steps, field_strength, seed, params, opt_state, instance
        )

        best_solution_found = []

        if i % 10 == 0:
            test_loss = jnp.mean(
                jnp.asarray(
                    [
                        jnp.min(run_sampler(sample_steps, seed, params, p)[1])
                        for p in test_dataset
                    ]
                )
            ).item()
            best_solution_found.append(test_loss)
            print("step %r loss train %r test %r", i, test_loss)
            # logging.info("step %r loss train %r test %r", step, train_loss, test_loss)
    return network, params
'''