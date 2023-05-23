import haiku as hk
import jax
import jraph
from jraph._src import utils
import jax
from python.src.sat_representations import SATRepresentation, VCG, LCG


def get_embedding(graph: jraph.GraphsTuple):
    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=jax.vmap(hk.Linear(output_size=32)),
        embed_node_fn=jax.vmap(hk.Linear(output_size=32)),
    )
    graph = embedding(graph)
    return graph


def apply_interaction(graph: jraph.GraphsTuple, num_message_passing_steps: int = 5):
    def mlp(dims):
        net = []
        for d in dims:
            net.extend([hk.Linear(d), jax.nn.relu])
        return hk.Sequential(net)

    @jax.vmap
    @jraph.concatenated_args
    def update_fn(features):
        # net = mlp([20, 20, 20])
        ln = hk.LayerNorm(axis=-1, param_axis=-1, create_scale=True, create_offset=True)
        net = mlp([100, 100])
        return ln(net(features))

    for _ in range(num_message_passing_steps):
        gn = jraph.InteractionNetwork(
            update_edge_fn=update_fn,
            update_node_fn=update_fn,
            include_sent_messages_in_node_update=True,
        )
        # update_edge_fn: a function mapping a single edge update inputs to a single edge feature.
        # update_node_fn: a function mapping a single node update input to a single node feature.
        # aggregate_edges_for_nodes_fn: function used to aggregate messages to each node.
        # include_sent_messages_in_node_update: pass edge features for which a node is a sender to the node update function.
        graph = gn(graph)
    return graph


def apply_convolution(graph: jraph.GraphsTuple, num_message_passing_steps: int = 5):
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
        gn = jraph.GraphConvolution(
            update_node_fn=update_fn,
            add_self_edges=False,
            aggregate_nodes_fn=utils.segment_sum,
        )
        # NOTE: implementation does not add an activation after aggregation; if we
        # stack layers, we might want to add an activation between each layer
        graph = gn(graph)
        # graph = graph._replace(nodes=jax.nn.relu(graph.nodes))
        # gn = jraph.GraphConvolution(
        #    update_node_fn=update_fn,
        # )
        # graph = gn(graph)
    return graph


def network_definition_interaction_VCG(
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
    graph = get_embedding(graph)
    graph = apply_interaction(graph, num_message_passing_steps)
    return hk.Linear(2)(graph.nodes)


def network_definition_interaction_LCG(
    graph: jraph.GraphsTuple,
    num_message_passing_steps: int = 5,
) -> jraph.ArrayTree:
    """Defines a graph neural network.
    Args:
      graph: Graphstuple the network processes.
      num_message_passing_steps: number of message passing steps.
    Returns:
      Decoded nodes.
    number_message_passing_steps = number of layers
    """
    graph = get_embedding(graph)
    graph = apply_interaction(graph, num_message_passing_steps)
    return hk.Linear(1)(graph.nodes)


def network_definition_convolution_VCG(
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
    graph = get_embedding(graph)
    graph = apply_convolution(graph, num_message_passing_steps)
    return hk.Linear(2)(graph.nodes)


def network_definition_convolution_LCG(
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
    graph = get_embedding(graph)
    graph = apply_convolution(graph, num_message_passing_steps)
    return hk.Linear(1)(graph.nodes)


def get_network_definition(network_type, graph_representation: SATRepresentation):
    match network_type, graph_representation:
        case "GCN", VCG:
            return network_definition_convolution_VCG
        case "GCN", LCG:
            return network_definition_convolution_LCG
        case "interaction", VCG:
            return network_definition_interaction_VCG
        case "interaction", LCG:
            return network_definition_interaction_LCG
        case _:
            raise ValueError("Invalid network_type or graph_representation")


'''
def network_definition_interaction(
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
        gn = jraph.InteractionNetwork(
            update_edge_fn=update_fn,
            update_node_fn=update_fn,
        )
        #    include_sent_messages_in_node_update=True,
        # )
        # update_edge_fn: a function mapping a single edge update inputs to a single edge feature.
        # update_node_fn: a function mapping a single node update input to a single node feature.
        # aggregate_edges_for_nodes_fn: function used to aggregate messages to each node.
        # include_sent_messages_in_node_update: pass edge features for which a node is a sender to the node update function.
        graph = gn(graph)
    return hk.Linear(2)(graph.nodes)


def network_definition_interaction_single_output(
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
        gn = jraph.InteractionNetwork(
            update_edge_fn=update_fn,
            update_node_fn=update_fn,
        )
        #    include_sent_messages_in_node_update=True,
        # )
        # update_edge_fn: a function mapping a single edge update inputs to a single edge feature.
        # update_node_fn: a function mapping a single node update input to a single node feature.
        # aggregate_edges_for_nodes_fn: function used to aggregate messages to each node.
        # include_sent_messages_in_node_update: pass edge features for which a node is a sender to the node update function.
        graph = gn(graph)
    # return jnp.array([jnp.sum(hk.Linear(3)(graph.nodes),axis=1)]).T
    return hk.Linear(1)(graph.nodes)


def network_definition_GCN(
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
        gn = jraph.GraphConvolution(
            update_node_fn=update_fn,
            add_self_edges=False,
            aggregate_nodes_fn=utils.segment_sum,
        )
        # NOTE: implementation does not add an activation after aggregation; if we
        # stack layers, we might want to add an activation between each layer
        # update_node_fn: function used to update the nodes. In the paper a single
        # layer MLP is used.
        # aggregate_nodes_fn: function used to aggregates the sender nodes.
        # add_self_edges: whether to add self edges to nodes in the graph as in the
        # paper definition of GCN. Defaults to False.
        # symmetric_normalization: whether to use symmetric normalization. Defaults
        # to True. Note that to replicate the fomula of the linked paper, the
        # adjacency matrix must be symmetric. If the adjacency matrix is not
        # symmetric the data is prenormalised by the sender degree matrix and post
        # normalised by the receiver degree matrix.
        graph = gn(graph)
        # graph = graph._replace(nodes=jax.nn.relu(graph.nodes))
        # gn = jraph.GraphConvolution(
        #    update_node_fn=update_fn,
        # )
        # graph = gn(graph)

    return hk.Linear(2)(graph.nodes)


def network_definition_GCN_single_output(
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
        gn = jraph.GraphConvolution(
            update_node_fn=update_fn,
            add_self_edges=False,
            aggregate_nodes_fn=utils.segment_sum,
        )
        # NOTE: implementation does not add an activation after aggregation; if we
        # stack layers, we might want to add an activation between each layer
        # update_node_fn: function used to update the nodes. In the paper a single
        # layer MLP is used.
        # aggregate_nodes_fn: function used to aggregates the sender nodes.
        # add_self_edges: whether to add self edges to nodes in the graph as in the
        # paper definition of GCN. Defaults to False.
        # symmetric_normalization: whether to use symmetric normalization. Defaults
        # to True. Note that to replicate the fomula of the linked paper, the
        # adjacency matrix must be symmetric. If the adjacency matrix is not
        # symmetric the data is prenormalised by the sender degree matrix and post
        # normalised by the receiver degree matrix.
        graph = gn(graph)
        # graph = graph._replace(nodes=jax.nn.relu(graph.nodes))
        # gn = jraph.GraphConvolution(
        #    update_node_fn=update_fn,
        # )
        # graph = gn(graph)
    # return jnp.array([jnp.sum(hk.Linear(3)(graph.nodes),axis=1)]).T
    return hk.Linear(1)(graph.nodes)
'''
