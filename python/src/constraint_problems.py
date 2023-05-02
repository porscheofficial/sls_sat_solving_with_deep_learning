"""
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
import scipy

# from pysat.solvers import Cadical

LabeledProblem = collections.namedtuple("Problem", ("graph", "labels", "mask", "meta"))

SATProblem = collections.namedtuple(
    "SATProblem", ("graph", "mask", "params", "clause_lengths")
)


class HashableSATProblem(SATProblem):
    def __hash__(self):
        return hash(
            (
                self.graph.senders.tostring(),
                self.graph.receivers.tostring(),
                self.graph.edges.tostring(),
                self.params,
                tuple(self.clause_lengths),
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


def get_problem_from_cnf(
    cnf: CNF, mode, pad_nodes=0, pad_edges=0
) -> HashableSATProblem:
    cnf.clauses = [c for c in cnf.clauses if len(c) > 0]
    n = cnf.nv
    # print("n=", n)
    m = len(cnf.clauses)
    # print("m=", m)
    # mode = "LCG"

    if mode == "VCG":
        # print("mode VCG")
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

        # For the loss calculation we create a mask for the nodes, which masks
        # the constraint nodes and the padding nodes.
        mask = (np.arange(n_node) < n).astype(np.int32)

        edges = np.eye(2)[edges]
        nodes = np.eye(2)[nodes]

    elif mode == "LCG":
        n_node = 2 * n + m
        clause_lengths = [len(c) for c in cnf.clauses]
        k = max(clause_lengths)
        n_edge = sum(clause_lengths) + n

        edges = []
        senders = []
        receivers = []

        # 1 indicates a literal node.
        # -1 indicated a negated literal node.
        # 0 indicates a constraint node.

        nodes = []
        for i in range(n_node):
            if i < 2 * n:
                if i % 2 == 0:
                    nodes.append(1)
                if i % 2 == 1:
                    nodes.append(-1)
            else:
                nodes.append(0)
        for j, c in enumerate(cnf.clauses):
            support = [(abs(l) - 1) for l in c]
            assert len(support) == len(
                set(support)
            ), "Multiple occurrences of single variable in constraint"

            # vals = ((np.sign(c) + 1) // 2).astype(np.int32)
            vals = ((np.sign(c))).astype(np.int32)
            for ii in range(len(vals)):
                if vals[ii] == 1:
                    senders.append(int(2 * support[ii] + 1))
                else:
                    senders.append(int(2 * support[ii]))
            edges.extend(np.repeat(0, len(c)))
            receivers.extend(np.repeat(j + 2 * n, len(c)))

        for jj in range(n):
            senders.append(int(2 * jj + 1))
            receivers.append(int(2 * jj))
            edges.append(1)

        assert len(nodes) == n_node
        assert len(receivers) == len(senders)
        assert len(senders) == len(edges)
        assert len(edges) == n_edge

        edges = np.eye(2)[edges]
        nodes = np.eye(2)[nodes]

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
    """
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

    edges = np.eye(2)[edges]
    """
    # this encodes edges between neighboring clauses
    """
    for j1, c1 in enumerate(cnf.clauses):
        for j2, c2 in enumerate(cnf.clauses):
            variables1 = [
                (abs(l1)) for l1 in c1
            ]  # gives the support qubits for clause c1
            variables2 = [
                (abs(l2)) for l2 in c2
            ]  # gives the support qubits for clause c2
            intersection = list(
                set(variables1) & set(variables2)
            )  # if this is non-empty, c1 and c2 are neighbors

            if len(intersection) != 0:
                if mode == "LCG":
                    senders.extend([j1 + 2 * n])
                    receivers.extend([j2 + 2 * n])
                if mode == "VCG":
                    senders.extend([j1 + n])
                    receivers.extend([j2 + n])
                edges = np.vstack(
                    (edges, [0, 0])
                )  # have to decide whether we give a weight here!

    n_edge = len(edges)

    assert len(receivers) == len(senders)
    assert len(senders) == len(edges)
    """
    # print("n,m", n, m)
    graph = jraph.GraphsTuple(
        n_node=np.asarray([n_node]),
        n_edge=np.asarray([n_edge]),
        edges=edges,
        nodes=nodes,
        globals=None,
        senders=np.asarray(senders),
        receivers=np.asarray(receivers),
    )

    if mode == "VCG":
        row_ind = np.asarray(senders)
        col_ind = np.asarray(receivers) - n * np.ones(len(receivers))
        data = np.ones(len(row_ind))
        # rint("n,m", n, m)
        sparse_clause_matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), (n, m)
        )
        # print(sparse_clause_matrix.shape)
        adj_matrix = sparse_clause_matrix.transpose() @ sparse_clause_matrix
        major_dimension, minor_dimension = adj_matrix.shape
        minor_indices = adj_matrix.indices
        major_indices = np.empty(len(minor_indices), dtype=adj_matrix.indices.dtype)
        scipy.sparse._sparsetools.expandptr(
            major_dimension, adj_matrix.indptr, major_indices
        )
        x, y = np.array(
            np.where(
                minor_indices - major_indices != 0,
                [minor_indices + n, major_indices + n],
                0,
            )
        )
        x = x[x != 0]
        y = y[y != 0]
        neighbors_list = np.vstack((y, x))
        """
        e = len(senders)
        data = np.where(np.tile(senders, e) == np.repeat(senders, e), 1, 0)
        x = data * np.tile(receivers, e)
        y = data * np.repeat(receivers, e)
        x, y = np.array(np.where(x - y != 0, [x, y], 0))
        x, y = np.unique(np.vstack((x, y)), axis=1)
        x = x[x != 0]
        y = y[y != 0]
        neighbors_list = np.stack((x, y))
        """
    if mode == "LCG":
        row_ind = np.floor(np.asarray(senders[:-n]) / 2)
        col_ind = np.asarray(receivers[:-n]) - 2 * n * np.ones(len(receivers[:-n]))
        data = np.ones(len(row_ind))
        # print("n,m", n, m)
        sparse_clause_matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), (n, m)
        )
        # print(sparse_clause_matrix.shape)
        adj_matrix = sparse_clause_matrix.transpose() @ sparse_clause_matrix
        major_dimension, minor_dimension = adj_matrix.shape
        minor_indices = adj_matrix.indices
        major_indices = np.empty(len(minor_indices), dtype=adj_matrix.indices.dtype)
        scipy.sparse._sparsetools.expandptr(
            major_dimension, adj_matrix.indptr, major_indices
        )
        x, y = np.array(
            np.where(
                minor_indices - major_indices != 0,
                [minor_indices + 2 * n, major_indices + 2 * n],
                0,
            )
        )
        x = x[x != 0]
        y = y[y != 0]
        neighbors_list = np.vstack((y, x))
        """
        # returns the wrong result
        new_senders = (np.array(senders)+1)%2* np.array(senders) + (np.array(senders))%2*(np.array(senders)-1)
        print(new_senders)
        e = len(new_senders)
        data = np.where(np.tile(new_senders, e) == np.repeat(new_senders, e),1, 0)
        x = data * np.tile(receivers, e)
        y = data * np.repeat(receivers, e)
        x,y = np.array(np.where(x-y!=0, [x,y],0))
        x,y = np.unique(np.vstack((x,y)),axis=1)
        x = x[x!=0]
        y = y[y!=0]
        neighbors_list = np.stack((x,y))
        """
        # with two for-loops
        """
        sender_clauses = []
        receiver_clauses = []
        for j1, c1 in enumerate(cnf.clauses):
            for j2, c2 in enumerate(cnf.clauses):
                variables1 = [
                    (abs(l1)) for l1 in c1
                ]  # gives the support qubits for clause c1
                variables2 = [
                    (abs(l2)) for l2 in c2
                ]  # gives the support qubits for clause c2
                intersection = list(
                    set(variables1) & set(variables2)
                )  # if this is non-empty, c1 and c2 are neighbors

                if len(intersection) != 0:
                    # if mode == "LCG":
                    sender_clauses.extend([j1 + 2 * n])
                    receiver_clauses.extend([j2 + 2 * n])
                    # if mode == "VCG":
                    #    sender_clauses.extend([j1 + n])
                    #    receiver_clauses.extend([j2 + n])
        neighbors_list2 = np.vstack(
            (np.array(sender_clauses), np.array(receiver_clauses))
        )

        print("difference", neighbors_list - neighbors_list2)
        """
    # alternative methods for computing the neighbors_list
    """
    if mode == "LCG":
        sender_clauses = []
        receiver_clauses = []
        for j1, c1 in enumerate(cnf.clauses):
            for j2, c2 in enumerate(cnf.clauses):
                variables1 = [
                    (abs(l1)) for l1 in c1
                ]  # gives the support qubits for clause c1
                variables2 = [
                    (abs(l2)) for l2 in c2
                ]  # gives the support qubits for clause c2
                intersection = list(
                    set(variables1) & set(variables2)
                )  # if this is non-empty, c1 and c2 are neighbors

                if len(intersection) != 0:
                    if mode == "LCG":
                        sender_clauses.extend([j1 + 2 * n])
                        receiver_clauses.extend([j2 + 2 * n])
                    if mode == "VCG":
                        sender_clauses.extend([j1 + n])
                        receiver_clauses.extend([j2 + n])
        neighbors_list = np.vstack((np.array(sender_clauses), np.array(receiver_clauses)))
    elif mode == "VCG":
        edges_matrix = np.sum(edges, axis = 1)
        adjacency_matrix = scipy.sparse.coo_matrix(
                        (
                            edges_matrix,
                            np.column_stack((np.asarray(receivers), np.asarray(senders))).T,
                        ),
                        shape=(n_node, n_node),
                    )
        adjacency_matrix = adjacency_matrix.todense()
        def get_neighborhood(matrix):
            sender_clauses = []
            receiver_clauses = []
            for i in range(np.shape(matrix)[1]):
                target = np.ravel(matrix[i,:])
                a = np.multiply(matrix, target)
                a = np.ravel(np.sum(a, axis = 1))
                a[i] = 0
                b = np.argwhere(a != 0)
                sender_clauses.extend(np.ones(len(b))*i)
                receiver_clauses.extend(b)
                # neighbors_indices = np.argwhere(b != 0).transpose()[0]
                # neighbors_indices_list.append(neighbors_indices)
            neighbors_indices_list = np.vstack((np.array(sender_clauses), np.ravel(np.array(receiver_clauses))))
            return neighbors_indices_list

        neighbors_list = get_neighborhood(
                        adjacency_matrix
                    )
    """
    # padding done in case we want to jit the graph, this is relevant mostly for training the gnn model, not for
    # executing moser's walk on single instances

    if pad_nodes > n_node or pad_edges > n_edge:
        n_node = max(pad_nodes, n_node)
        n_edge = max(pad_edges, n_edge)
        graph = jraph.pad_with_graphs(graph, n_node, n_edge)
        # neighbors_list =  np.pad(neighbors_list, ((0,n_node - np.shape(neighbors_list)[0]),(0,np.shape(neighbors_list)[0])))

    # For the loss calculation we create a mask for the nodes, which masks
    # the constraint nodes and the padding nodes.
    if mode == "LCG":
        # mask = (np.arange(n_node) < n).astype(np.int32)
        mask = (np.arange(n_node) < 2 * n).astype(np.int32)

    elif mode == "VCG":
        mask = (np.arange(n_node) < n).astype(np.int32)

    assert len(mask) == n_node

    return HashableSATProblem(
        graph=graph,
        mask=[mask, neighbors_list],
        clause_lengths=clause_lengths,
        params=(n, m, k),
    )


# def construct_constraint_graph(graph: jraph.GraphsTuple) -> jraph:GraphsTuple:
#     e = len(graph.edges)
#     n = graph.n_node
#     adjacency_matrix = BCOO(
#             (
#                 np.ones(e),
#                 np.column_stack((graph.senders, graph.receivers)),
#             ),
#             shape=(n, n),
#         )
#     # two hop adjacency matrix with values indicating number of paths.
#     adj_squared =  full_adjacency_matrix @ full_adjacency_matrix
#     return adj_squared.unique_indices

"""
def get_solved_problem_from_cnf(cnf: CNF, solver=Cadical()):
    solver.append_formula(cnf.clauses)
    solution_found = solver.solve()
    solution = None
    if solution_found:
        solution = solver.get_model()
    return get_problem_from_cnf(cnf, solution)
"""