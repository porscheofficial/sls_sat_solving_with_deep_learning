import collections
import jraph
import numpy as np
from pysat.formula import CNF
from python.src.sat_representations import SATRepresentation

SATProblem = collections.namedtuple(
    "SATProblem", ("graph", "mask", "constraint_graph", "params", "constraint_mask")
)


class HashableSATProblem(SATProblem):
    def __hash__(self):
        return hash(
            (
                self.graph.senders.tostring(),
                self.graph.receivers.tostring(),
                self.graph.edges.tostring(),
                self.params,
            )
        )

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def all_bitstrings(size):
    bitstrings = np.ndarray((2**size, size), dtype=int)
    for i in range(size):
        bitstrings[:, i] = np.tile(
            np.repeat(np.array([0, 1]), 2 ** (size - i - 1)), 2**i
        )
    return bitstrings


def get_problem_from_cnf(
    cnf: CNF,
    representation: SATRepresentation,
    include_constraint_graph=False,
    pad_nodes=0,
    pad_edges=0,
) -> HashableSATProblem:
    clauses = [c for c in cnf.clauses if len(c) > 0]
    n = cnf.nv
    m = len(clauses)
    clause_lengths = [len(c) for c in clauses]
    k = max(clause_lengths)

    nodes, senders, receivers, edges, n_node, n_edge = representation.get_graph(
        n, m, clauses, clause_lengths
    )

    assert len(nodes) == n_node
    assert len(receivers) == len(senders)
    assert len(senders) == len(edges)
    assert len(edges) == n_edge

    graph = jraph.GraphsTuple(
        n_node=np.asarray([n_node]),
        n_edge=np.asarray([n_edge]),
        edges=np.eye(2)[edges],
        nodes=np.eye(2)[nodes],
        globals=None,
        senders=np.asarray(senders),
        receivers=np.asarray(receivers),
    )

    # constraint graph
    constraint_graph = (
        representation.get_constraint_graph(n, m, senders, receivers)
        if include_constraint_graph
        else None
    )
    constraint_mask = (
        np.array(np.logical_not(representation.get_mask(n, n_node)), dtype=int)
        if include_constraint_graph
        else None
    )

    # padding
    if pad_nodes > n_node or pad_edges > n_edge:
        n_node = max(pad_nodes, n_node)
        n_edge = max(pad_edges, n_edge)
        graph = jraph.pad_with_graphs(graph, n_node, n_edge)

    if include_constraint_graph:
        constraint_graph = jraph.pad_with_graphs(
            constraint_graph, n_node, constraint_graph.n_edge
        )
        constraint_mask = np.pad(constraint_mask, (0, n_node - len(constraint_mask)))

    # mask
    mask = representation.get_mask(n, n_node).astype(np.int32)

    return HashableSATProblem(
        graph=graph,
        mask=mask,
        constraint_mask=constraint_mask,
        constraint_graph=constraint_graph,
        params=(n, m, k),
    )
