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
