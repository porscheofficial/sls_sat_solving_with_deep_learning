r"""
TODO: COnsider that this was originally taken from jraph examples, so check the license

We represent this problem in form of a bipartite-graph, with edges
    connecting the literal-nodes (a, b, c) with the constraint-nodes (O).
The corresponding graph looks like this:
O    O   O
|\  /\  /|
| \/  \/ |
| /\  /\ |
|/  \/  \|
a    b   c
The nodes are one-hot encoded with literal nodes as (1, 0) and constraint nodes
as (0, 1). The edges are one-hot encoded with (1, 0) if the literal should be
true and (0, 1) if the literal should be false.
The graph neural network encodes the nodes and the edges and runs multiple
message passing steps by calculating message for each edge and aggregating
    all the messages of the nodes.
The training dataset consists of randomly generated 2-sat problems with 2 to 15
literals.
The test dataset consists of randomly generated 2-sat problems with 16 to 20
literals.
"""

import collections
import jraph
import numpy as np
import random
from pysat.formula import CNF
from pysat.solvers import Cadical

LabeledProblem = collections.namedtuple("Problem", ("graph", "labels", "mask", "meta"))

SATProblem = collections.namedtuple("SATProblem", ("graph", "mask", "params", "clause_lengths"))


class HashableSATProblem(SATProblem):
    def __hash__(self):
        return hash((self.graph.senders.tostring(),
                     self.graph.receivers.tostring(),
                     self.graph.edges.tostring(),
                     self.params,
                     tuple(self.clause_lengths)
                     ))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()


def all_bitstrings(size):
    bitstrings = np.ndarray((2 ** size, size), dtype=int)
    for i in range(size):
        bitstrings[:, i] = np.tile(
            np.repeat(np.array([0, 1]), 2 ** (size - i - 1)), 2 ** i
        )
    return bitstrings


# @TODO: Update to edge mask
def get_2sat_problem(min_n_literals: int, max_n_literals: int) -> LabeledProblem:
    """Creates bipartite-graph representing a randomly generated 2-sat problem.
    Args:
    min_n_literals: minimum number of literals in the 2-sat problem.
    max_n_literals: maximum number of literals in the 2-sat problem.
    Returns:
    bipartite-graph, node labels and node mask.
    """
    n_literals = random.randint(min_n_literals, max_n_literals)
    n_literals_true = random.randint(1, n_literals - 1)
    n_constraints = n_literals * (n_literals - 1) // 2

    n_node = n_literals + n_constraints
    # 0 indicates a literal node
    # 1 indicates a constraint node.
    nodes = [0 if i < n_literals else 1 for i in range(n_node)]
    edges = []
    senders = []
    for literal_node1 in range(n_literals):
        for literal_node2 in range(literal_node1 + 1, n_literals):
            senders.append(literal_node1)
            senders.append(literal_node2)
            # 1 indicates that the literal must be true for this constraint.
            # 0 indicates that the literal must be false for this constraint.
            # I.e. with literals a and b, we have the following possible constraints:
            # 0, 0 -> a or b
            # 1, 0 -> not a or b
            # 0, 1 -> a or not b
            # 1, 1 -> not a or not b
            edges.append(1 if literal_node1 < n_literals_true else 0)
            edges.append(1 if literal_node2 < n_literals_true else 0)

    graph = jraph.GraphsTuple(
        n_node=np.asarray([n_node]),
        n_edge=np.asarray([2 * n_constraints]),
        # One-hot encoding for nodes and edges.
        edges=np.eye(2)[edges],
        nodes=np.eye(2)[nodes],
        globals=None,
        senders=np.asarray(senders),
        receivers=np.repeat(np.arange(n_constraints) + n_literals, 2),
    )

    # In order to jit compile our code, we have to pad the nodes and edges of
    # the GraphsTuple to a static shape.
    max_n_constraints = max_n_literals * (max_n_literals - 1) // 2
    max_nodes = max_n_literals + max_n_constraints + 1
    max_edges = 2 * max_n_constraints
    graph = jraph.pad_with_graphs(graph, max_nodes, max_edges)

    # The ground truth solution for the 2-sat problem.
    labels = (np.arange(max_nodes) < n_literals_true).astype(np.int32)
    labels = np.eye(2)[labels]

    # For the loss calculation we create a mask for the nodes, which masks
    # the constraint nodes and the padding nodes.
    mask = (np.arange(max_nodes) < n_literals).astype(np.int32)
    meta = {"n_vars": n_literals, "n_constraints": n_constraints}
    return SATProblem(graph=graph, labels=labels, mask=mask, meta=meta)


# @TODO:Update to edge mask
def get_k_sat_problem(n, m, k):
    n_node = n + m
    nodes = [0 if i < n else 1 for i in range(n_node)]
    edges = []
    senders = []
    receivers = []

    for c in range(n, n_node):
        support = np.random.choice(n, replace=False, size=(k,))
        bits = np.random.randint(2, size=(k,))
        senders.extend(support)
        edges.extend(bits)
        receivers.extend(np.repeat(c, k))

    graph = jraph.GraphsTuple(
        n_node=np.asarray([n_node]),
        n_edge=np.asarray([m * k]),
        edges=np.eye(2)[edges],
        nodes=np.eye(2)[nodes],
        globals=None,
        senders=np.asarray(senders),
        receivers=np.asarray(receivers),
    )

    # For the loss calculation we create a mask for the nodes, which masks
    # the constraint nodes and the padding nodes.
    mask = (np.arange(n_node) < n).astype(np.int32)
    meta = {"n": n, "m": m, "k": k}

    # finally, we create the labels if any have been fed
    return SATProblem(graph=graph, mask=mask, meta=meta)


def get_problem_from_cnf(cnf: CNF, pad_nodes=0, pad_edges=0):
    cnf.clauses = [c for c in cnf.clauses if len(c) > 0]
    n = cnf.nv
    m = len(cnf.clauses)
    n_node = n + m
    clause_lengths = [len(c) for c in cnf.clauses]
    k = max(clause_lengths)
    n_edge = sum(clause_lengths)

    # for sake of jitting, if the cnf isn't already strictly in k-cnf form, we introduce
    # additional dummy variables and constraints. NB: While this in principles solves the problem,
    # it actually is to be avoided, if possible: This is because it very easy to satisfy all constraint except one
    # by just setting the dummy variables to True. This creates local minima and also breaks locality.
    # if any([len(c) != k for c in cnf.clauses]):
    #     m += 2 ** k - 1
    #     n += k
    #
    #     dummy_vars = np.arange(n - k, n)
    #     senders.extend(np.repeat(dummy_vars, 2 ** k - 1))
    #
    #     # we introduce additional constraints to force the dummy variables into the all zeros string
    #     additional_constraints = all_bitstrings(k)[1:, :]
    #
    #     for j in range(2 ** k - 1):
    #         edges.extend(additional_constraints[j, :])
    #         receivers.extend(np.repeat(m - 2 ** k + 1, k))

    edges = []
    senders = []
    receivers = []
    nodes = [0 if i < n else 1 for i in range(n_node)]
    for j, c in enumerate(cnf.clauses):
        support = [(abs(l) - 1) for l in c]
        assert len(support) == len(
            set(support)
        ), "Multiple occurrences of single variable in constraint"

        vals = ((np.sign(c) + 1) // 2).astype(np.int32)

        senders.extend(support)
        edges.extend(vals)
        receivers.extend(np.repeat(j + n, len(c)))

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

    # padding done in case we want to jit the graph, this is relevant mostly for training the gnn model, not for
    # executing moser's walk on single instances

    if pad_nodes > n_node or pad_edges > n_edge:
        n_node = max(pad_nodes, n_node)
        n_edge = max(pad_edges, n_edge)
        graph = jraph.pad_with_graphs(
            graph, n_node, n_edge
        )

    # For the loss calculation we create a mask for the nodes, which masks
    # the constraint nodes and the padding nodes.

    mask = (np.arange(pad_nodes) < n).astype(np.int32)

    return HashableSATProblem(
        graph=graph,
        mask=mask,
        clause_lengths=clause_lengths,
        params=(n, m, k)
    )


def get_solved_problem_from_cnf(cnf: CNF, solver=Cadical()):
    solver.append_formula(cnf.clauses)
    solution_found = solver.solve()
    solution = None
    if solution_found:
        solution = solver.get_model()
    return get_problem_from_cnf(cnf, solution)
