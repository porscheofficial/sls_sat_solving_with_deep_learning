"""Definition of Graph Neural Network (GNN) models."""
import haiku as hk
import jax
import jraph
from jraph._src import utils
from python.src.sat_representations import VCG, LCG

# from typing import Any


def get_embedding(graph: jraph.GraphsTuple):
    """Get embedded graph."""
    embedding = jraph.GraphMapFeatures(
        embed_edge_fn=jax.vmap(hk.Linear(output_size=32)),
        embed_node_fn=jax.vmap(hk.Linear(output_size=32)),
    )
    graph = embedding(graph)
    return graph


def apply_interaction(
    mlp_layers: list[int],
    graph: jraph.GraphsTuple,
    num_message_passing_steps: int = 5,
):
    """Apply an interaction net on a graph.

    Args:
        mlp_layers (list[int]): mlp_layer dimensions
        graph (jraph.GraphsTuple): graph we want to apply an interaction net on
        num_message_passing_steps (int, optional): number of message passing steps = number of layers. Defaults to 5.

    Returns:
        graph after application of interaction.
    """

    def mlp(dims):
        """Define an MLP."""
        net = []
        for dim in dims:
            net.extend([hk.Linear(dim), jax.nn.relu])
        return hk.Sequential(net)

    @jax.vmap
    @jraph.concatenated_args
    def update_fn(features):
        """Define an update function including a layer norm feature."""
        layer_norm = hk.LayerNorm(
            axis=-1, param_axis=-1, create_scale=True, create_offset=True
        )
        net = mlp(mlp_layers)
        return layer_norm(net(features))

    for _ in range(num_message_passing_steps):
        interaction_layer = jraph.InteractionNetwork(
            update_edge_fn=update_fn,
            update_node_fn=update_fn,
            include_sent_messages_in_node_update=True,
        )
        graph = interaction_layer(graph)
    return graph


def apply_convolution(
    mlp_layers: list[int], graph: jraph.GraphsTuple, num_message_passing_steps: int = 5
):
    """Apply a convolution net on a graph.

    Args:
        mlp_layers (list[int]): mlp_layer dimensions
        graph (jraph.GraphsTuple): graph we want to apply a convoution net on
        num_message_passing_steps (int, optional): number of message passing steps = number of layers. Defaults to 5.

    Returns:
        graph after convolution.
    """

    def mlp(dims):
        """Define an MLP."""
        net = []
        for dim in dims:
            net.extend([hk.Linear(dim), jax.nn.relu])
        return hk.Sequential(net)

    @jax.vmap
    @jraph.concatenated_args
    def update_fn(features):
        """Define an update function."""
        net = mlp(mlp_layers)
        return net(features)

    for _ in range(num_message_passing_steps):
        convolution_layer = jraph.GraphConvolution(
            update_node_fn=update_fn,
            add_self_edges=False,
            aggregate_nodes_fn=utils.segment_sum,
        )
        # NOTE: implementation does not add an activation after aggregation; if we
        # stack layers, we might want to add an activation between each layer
        graph = convolution_layer(graph)
        # graph = graph._replace(nodes=jax.nn.relu(graph.nodes))
        # gn = jraph.GraphConvolution(
        #    update_node_fn=update_fn,
        # )
        # graph = gn(graph)
    return graph


def network_definition_interaction_vcg(
    graph: jraph.GraphsTuple, mlp_layers: list[int], num_message_passing_steps: int = 5
) -> jraph.ArrayTree:
    """Define a graph neural network (GNN) for interaction net and VCG.

    Args:
        graph (jraph.GraphsTuple): Graphstuple the network processes.
        mlp_layers (list[int]): dimension of mlp layers, e.g. [200,200]
        num_message_passing_steps (int, optional): number of message passing steps = number of layers.

    Returns:
        jraph.ArrayTree: nodes after application of the net
    """
    graph = get_embedding(graph)
    graph = apply_interaction(mlp_layers, graph, num_message_passing_steps)
    return hk.Linear(2)(graph.nodes)


def network_definition_interaction_lcg(
    graph: jraph.GraphsTuple,
    mlp_layers: list[int],
    num_message_passing_steps: int = 5,
) -> jraph.ArrayTree:
    """Define a graph neural network (GNN) for interaction net and LCG.

    Args:
        graph (jraph.GraphsTuple): Graphstuple the network processes.
        mlp_layers (list[int]): dimension of mlp layers, e.g. [200,200]
        num_message_passing_steps (int, optional): number of message passing steps = number of layers. Defaults to 5.

    Returns:
        jraph.ArrayTree: nodes after application of the net
    """
    graph = get_embedding(graph)
    graph = apply_interaction(mlp_layers, graph, num_message_passing_steps)
    return hk.Linear(1)(graph.nodes)


def network_definition_convolution_vcg(
    graph: jraph.GraphsTuple, mlp_layers: list[int], num_message_passing_steps: int = 5
) -> jraph.ArrayTree:
    """Define a graph neural network (GNN) for GCN and VCG.

    Args:
        graph (jraph.GraphsTuple): Graphstuple the network processes.
        mlp_layers (list[int]): dimension of mlp layers, e.g. [200,200]
        num_message_passing_steps (int, optional): number of message passing steps = number of layers. Defaults to 5.

    Returns:
        jraph.ArrayTree: nodes after application of the net
    """
    graph = get_embedding(graph)
    graph = apply_convolution(mlp_layers, graph, num_message_passing_steps)
    return hk.Linear(2)(graph.nodes)


def network_definition_convolution_lcg(
    graph: jraph.GraphsTuple, mlp_layers: list[int], num_message_passing_steps: int = 5
) -> jraph.ArrayTree:
    """Define a graph neural network (GNN) for GCN and LCG.

    Args:
        graph (jraph.GraphsTuple): Graphstuple the network processes.
        mlp_layers (list[int]): dimension of mlp layers, e.g. [200,200]
        num_message_passing_steps (int, optional): number of message passing steps = number of layers. Defaults to 5.

    Returns:
        jraph.ArrayTree: nodes after application of the net
    """
    graph = get_embedding(graph)
    graph = apply_convolution(mlp_layers, graph, num_message_passing_steps)
    return hk.Linear(1)(graph.nodes)


def get_network_definition(network_type, graph_representation):
    """Get proper network definition from (network_type, graph_representation).

    Args:
        network_type (str): either "GCN" or "interaction". Note: only "interaction" is properly tested...
        graph_representation (SATRepresentation): Graph representation -> either LCG or VCG.

    Raises:
        ValueError: if something is chosen that is not implemented in this framework

    Returns:
        returns network definition for the chosen input
    """
    if network_type == "GCN" and graph_representation == VCG:
        function = network_definition_convolution_vcg
    elif network_type == "GCN" and graph_representation == LCG:
        function = network_definition_convolution_lcg
    elif network_type == "interaction" and graph_representation == VCG:
        function = network_definition_interaction_vcg
    elif network_type == "interaction" and graph_representation == LCG:
        function = network_definition_interaction_lcg
    else:
        raise ValueError("Invalid network_type or graph_representation")
    return function
