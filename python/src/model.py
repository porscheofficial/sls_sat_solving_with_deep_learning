import haiku as hk
import jax
import jraph
from jraph._src import utils
import jax.numpy as jnp

from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np


def get_embedding(graph: jraph.GraphsTuple):
    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=jax.vmap(hk.Linear(output_size=16)),
        embed_node_fn=jax.vmap(hk.Linear(output_size=16)),
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
        net = mlp([20, 20, 20])
        # net = mlp([40, 40, 40])
        return net(features)

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


def get_model_probabilities(network, params, problem, mode):
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
    # mode = "LCG"
    n, _, _ = problem.params
    decoded_nodes = network.apply(params, problem.graph)
    if mode == "VCG":
        return jax.nn.softmax(decoded_nodes)[:n, 1]
    if mode == "LCG":
        if np.shape(decoded_nodes)[0] % 2 == 1:
            decoded_nodes = jnp.vstack((jnp.asarray(decoded_nodes), [[0]]))
            conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        else:
            conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        return jax.nn.softmax(conc_decoded_nodes)[:n, 1]


def get_network_definition(network_type, graph_representation):
    if network_type == "GCN" and graph_representation == "VCG":
        network_definition = network_definition_convolution_VCG
    elif network_type == "GCN" and graph_representation == "LCG":
        network_definition = network_definition_convolution_LCG
    elif network_type == "interaction" and graph_representation == "VCG":
        network_definition = network_definition_interaction_VCG
    elif network_type == "interaction" and graph_representation == "LCG":
        network_definition = network_definition_interaction_LCG
    else:
        print("Network not defined. Please use a type that is defined")
    return network_definition


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
