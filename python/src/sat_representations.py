from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import jax
from jraph._src import utils
from jax.experimental.sparse import BCOO
from functools import partial
from pysat.formula import CNF
import scipy


class SATRepresentation(ABC):
    @staticmethod
    @abstractmethod
    def get_graph(n, m, clauses, clause_lengths):
        pass

    @staticmethod
    @abstractmethod
    def get_constraint_graph(n, m, senders, receivers):
        pass

    @staticmethod
    @abstractmethod
    def get_violated_constraints(problem, assignment):
        pass

    @staticmethod
    @abstractmethod
    def get_mask(n, n_node):
        pass

    @staticmethod
    @abstractmethod
    def prediction_loss():
        pass

    @staticmethod
    @abstractmethod
    def lll_loss():
        pass

    @staticmethod
    @abstractmethod
    def get_n_nodes(cnf: CNF):
        pass

    @staticmethod
    @abstractmethod
    def get_n_edges(cnf: CNF):
        pass


class VCG(SATRepresentation):
    @staticmethod
    @abstractmethod
    def get_n_nodes(cnf: CNF):
        n = cnf.nv
        m = len(cnf.clauses)
        return n + m

    @staticmethod
    @abstractmethod
    def get_n_edges(cnf: CNF):
        return sum([len(c) for c in cnf.clauses])

    @staticmethod
    def get_graph(n, m, clauses, clause_lengths):
        n_node = n + m
        n_edge = sum(clause_lengths)

        edges = []
        senders = []
        receivers = []
        nodes = [0 if i < n else 1 for i in range(n_node)]
        for j, c in enumerate(clauses):
            support = [(abs(l) - 1) for l in c]
            assert len(support) == len(
                set(support)
            ), "Multiple occurrences of single variable in constraint"

            vals = ((np.sign(c) + 1) // 2).astype(np.int32)

            senders.extend(support)
            edges.extend(vals)
            receivers.extend(np.repeat(j + n, len(c)))

        return nodes, senders, receivers, edges, n_node, n_edge

    @staticmethod
    def get_constraint_graph(n, m, senders, receivers):
        row_ind = np.asarray(senders)
        col_ind = np.asarray(receivers) - n * np.ones(len(receivers))
        data = np.ones(len(row_ind))
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
        return neighbors_list

    @staticmethod
    def get_violated_constraints(problem, assignment):
        # @partial(jax.jit, static_argnames=("problem",))
        def violated_constraints(problem, assignment):
            graph = problem.graph
            edge_is_violated = jnp.mod(
                graph.edges[:, 1] + assignment.T[graph.senders].T, 2
            )

            e = len(graph.edges)
            _, m, k = problem.params
            edge_mask_sp = BCOO(
                (np.ones(e), np.column_stack((np.arange(e), graph.receivers))),
                shape=(e, m),
            )

            violated_constraint_edges = (
                edge_is_violated @ edge_mask_sp
            )  # (,x) @ (x,m)  = (,m)
            constraint_is_violated = violated_constraint_edges == jnp.asarray(
                problem.clause_lengths
            )

            return constraint_is_violated

        return np.sum(violated_constraints(problem, assignment).astype(int), axis=0)

    @staticmethod
    def get_mask(n, n_node):
        return np.arange(n_node) < n


class LCG(SATRepresentation):
    @staticmethod
    @abstractmethod
    def get_n_nodes(cnf: CNF):
        n = cnf.nv
        m = len(cnf.clauses)
        return 2 * n + m

    @staticmethod
    @abstractmethod
    def get_n_edges(cnf: CNF):
        """
        In LCG, there is one edge for each literal in each clause, plus one
        edge for each variable that connects the positive and negative literal
        node.
        """
        return sum([len(c) for c in cnf.clauses]) + cnf.nv

    @staticmethod
    def get_graph(n, m, clauses, clause_lengths):
        n_node = 2 * n + m
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
        for j, c in enumerate(clauses):
            support = [(abs(l) - 1) for l in c]
            assert len(support) == len(
                set(support)
            ), "Multiple occurrences of single variable in constraint"

            vals = ((np.sign(c) + 1) // 2).astype(np.int32)

            for i, val in enumerate(vals):
                if val == 1:
                    senders.append(int(2 * support[i] + 1))
                else:
                    senders.append(int(2 * support[i]))
            edges.extend(np.repeat(0, len(c)))
            receivers.extend(np.repeat(j + 2 * n, len(c)))

        for jj in range(n):
            senders.append(int(2 * jj + 1))
            receivers.append(int(2 * jj))
            edges.append(1)

        return nodes, senders, receivers, edges, n_node, n_edge

    @staticmethod
    def get_constraint_graph(n, m, senders, receivers):
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
        return neighbors_list

    # @partial(jax.jit, static_argnames=("problem",))
    @staticmethod
    def get_violated_constraints(problem, assignment):
        def one_hot(x, k, dtype=jnp.float32):
            """Create a one-hot encoding of x of size k."""
            return jnp.array(x[:, None] == jnp.arange(k), dtype)

        graph = problem.graph
        n, m, k = problem.params
        senders = graph.senders[:-n]
        receivers = graph.receivers[:-n]
        new_assignment = jnp.ravel(one_hot(assignment, 2))
        edge_is_satisfied = jnp.ravel(
            new_assignment[None].T[senders].T
        )  # + np.ones(len(senders)), 2)
        number_of_literals_satisfied = utils.segment_sum(
            data=edge_is_satisfied, segment_ids=receivers, num_segments=2 * n + m
        )[2 * n :]
        clause_is_unsat = jnp.where(number_of_literals_satisfied > 0, 0, 1)
        return jnp.sum(clause_is_unsat)

    @staticmethod
    def get_mask(n, n_node):
        return np.arange(n_node) < 2 * n
