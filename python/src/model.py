import haiku as hk
import jax
import jraph


def network_definition(
    graph: jraph.GraphsTuple, num_message_passing_steps: int = 5
) -> jraph.ArrayTree:
    """Defines a graph neural network.
    Args:
      graph: Graphstuple the network processes.
      num_message_passing_steps: number of message passing steps.
    Returns:
      Decoded nodes.
    number_message_passing_steps = number of layers
    """
    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=jax.vmap(hk.Linear(output_size=16)),
        embed_node_fn=jax.vmap(hk.Linear(output_size=16)),
    )
    # embedding = jraph.GraphMapFeatures(
    #    embed_edge_fn=jax.vmap(hk.Linear(output_size=32)),
    #    embed_node_fn=jax.vmap(hk.Linear(output_size=32)),
    # )

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
        # net = mlp([40, 40, 40])
        return net(features)

    for _ in range(num_message_passing_steps):
        # gn = jraph.InteractionNetwork(
        #    update_edge_fn=update_fn,
        #   update_node_fn=update_fn,
        #    include_sent_messages_in_node_update=True,
        # )
        # graph = gn(graph)
        gn = jraph.GraphConvolution(
            update_node_fn=update_fn,
        )
        graph = gn(graph)
        graph = graph._replace(nodes=jax.nn.relu(graph.nodes))
        gn = jraph.GraphConvolution(
            update_node_fn=update_fn,
        )

        graph = gn(graph)

    return hk.Linear(2)(graph.nodes)


def get_model_probabilities(network, params, problem):
    """
    Helper method that returns, for each, problem variable, the Bernoulli parameter of the model for this variable.
    That is, the ith value of the returned array is the probability with which the model will assign 1 to the
    ith variable.

    The reasoning for choosing the first, rather than the zeroth, column of the model output below is as follows:

    - When evaluating the loss function, candidates are one-hot encoded, which means that when a satisfying assignment
    for a problem sets variable i to 1, then this will increase the likelihood that the model will set this variable to
    1, meaning, all else being equal, a larger Bernoulli weight in element [i,1] of the model output. As a result the
    right column of the softmax of the model output equals the models likelihood for setting variables to 1, which is
    what we seek.
    """
    n, _, _ = problem.params
    decoded_nodes = network.apply(params, problem.graph)
    return jax.nn.softmax(decoded_nodes)[:n, 1]
