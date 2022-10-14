import jraph
import jax
import haiku as hk
import optax
from functools import partial
import jax.numpy as jnp
import numpy as np

from random_walk import moser_walk_sampler


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

    @jax.vmap
    @jraph.concatenated_args
    def update_fn(features):
        net = hk.Sequential(
            [
                hk.Linear(10),
                jax.nn.relu,
                hk.Linear(10),
                jax.nn.relu,
                hk.Linear(10),
                jax.nn.relu,
            ]
        )
        return net(features)

    for _ in range(num_message_passing_steps):
        gn = jraph.InteractionNetwork(
            update_edge_fn=update_fn,
            update_node_fn=update_fn,
            include_sent_messages_in_node_update=True,
        )
        graph = gn(graph)

    return hk.Linear(2)(graph.nodes)


def train_model(
    # num_steps: int,
    sample_steps: int,
    field_strength: float,
    train_dataset,
    test_dataset,
):
    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), train_dataset[0].graph)

    opt_init, opt_update = optax.adam(2e-4)
    opt_state = opt_init(params)

    @jax.jit
    def supervised_prediction_loss(ps, problem):
        decoded_nodes = network.apply(ps, problem.graph)
        # We interpret the decoded nodes as a pair of logits for each node.
        log_prob = jax.nn.log_softmax(decoded_nodes) * problem.labels
        return -jnp.sum(log_prob * problem.mask[:, None]) / jnp.sum(problem.mask)

    # @partial(jax.jit, static_argnames=("s", "f"))
    def unsupervised_prediction_loss(s: int, f: float, ps, problem):
        logits = network.apply(ps, problem.graph)

        # here we feed the model the second column of the output, since that is the probability that a "1" assignment
        # for each variable should be drawn
        model_probs = jax.nn.softmax(logits)[: problem.meta["n"]][:, 1]
        trajectory, energies = moser_walk_sampler(model_probs, problem, s)
        one_hot_encoded_trajectories = jnp.eye(2)[trajectory.astype(dtype=np.int32)]
        trajectory_log_probs = jnp.tensordot(
            one_hot_encoded_trajectories,
            jax.nn.log_softmax(logits)[: problem.meta["n"]],
            2,
        )
        unnormalized_weights = jnp.exp(-f * energies)
        weighted_log_probs = jnp.dot(trajectory_log_probs, unnormalized_weights)
        return -weighted_log_probs / (
            jnp.sum(unnormalized_weights) * jnp.sum(problem.mask)
        )

    # @partial(jax.jit, static_argnums=(0,1))
    def update(s, f, ps, opt, problem):
        loss = partial(unsupervised_prediction_loss, s, f)
        g = jax.grad(loss)(ps, problem)
        updates, opt = opt_update(g, opt)
        return optax.apply_updates(params, updates), opt

    for i in range(50):
        instance = train_dataset[0]
        # upd = partial(update, samples=sample_steps, field_strength)
        params, opt_state = update(
            sample_steps, field_strength, params, opt_state, instance
        )
        if i % 10 == 0:
            test_loss = jnp.mean(
                jnp.asarray(
                    [
                        unsupervised_prediction_loss(
                            sample_steps, field_strength, params, p
                        )
                        for p in test_dataset
                    ]
                )
            ).item()
            print("step %r loss train %r test %r", i, test_loss)
            # logging.info("step %r loss train %r test %r", step, train_loss, test_loss)
    return network, params
