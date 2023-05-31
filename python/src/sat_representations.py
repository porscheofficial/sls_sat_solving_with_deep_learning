from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import jax
from jraph._src import utils
from jax.experimental.sparse import BCOO
from functools import partial
from pysat.formula import CNF
import scipy
import jraph


class SATRepresentation(ABC):
    @staticmethod
    @abstractmethod
    def get_graph(n, m, clauses, clause_lengths):
        pass

    @staticmethod
    @abstractmethod
    def get_constraint_graph(n, m, senders, receivers) -> jraph.GraphsTuple:
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
    def get_n_nodes(cnf: CNF):
        pass

    @staticmethod
    @abstractmethod
    def get_n_edges(cnf: CNF):
        pass

    @staticmethod
    @abstractmethod
    def get_padded_candidate(solution_dict, n_nodes):
        pass

    @staticmethod
    @abstractmethod
    def get_model_probabilities(decoded_nodes, n):
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
        pass

    @staticmethod
    @abstractmethod
    def prediction_loss(decoded_nodes, mask, candidates, energies, inv_temp: float):
        pass

    @staticmethod
    @abstractmethod
    def local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        pass

    @staticmethod
    @abstractmethod
    def entropy_loss(decoded_nodes, mask):
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
    @abstractmethod
    def get_padded_candidate(solution_dict, n_nodes):
        #
        return np.pad(
            solution_dict,
            pad_width=(
                (0, 0),
                (0, int(np.ceil(n_nodes)) - np.shape(solution_dict)[1]),
            ),  # @TODO: check if this is correct
        )

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
    def get_constraint_graph(n, m, senders, receivers) -> jraph.GraphsTuple:
        row_ind = np.asarray(senders)
        col_ind = np.asarray(receivers) - n * np.ones(len(receivers))
        data = np.ones(len(row_ind))
        sparse_clause_matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), (n, m)
        )
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
        graph = jraph.GraphsTuple(
            n_node=np.asarray([m]),
            n_edge=np.asarray([len(x)]),
            senders=neighbors_list[0],
            receivers=neighbors_list[1],
            globals=None,
            edges=None,
            nodes=None,
        )
        return graph

    @staticmethod
    # @partial(jax.jit, static_argnames=("problem",))
    def get_violated_constraints(problem, assignment):
        graph = problem.graph
        n, m, _ = problem.params
        edge_is_satisfied = graph.edges[:, 1] == assignment.T[graph.senders].T
        number_of_literals_satisfied = utils.segment_sum(
            data=edge_is_satisfied.astype(int),
            segment_ids=graph.receivers - n,
            num_segments=m,
        )
        return jnp.where(number_of_literals_satisfied > 0, 0, 1)

    @staticmethod
    def get_mask(n, n_node):
        return np.arange(n_node) < n

    @staticmethod
    def get_model_probabilities(decoded_nodes, n):
        return jax.nn.softmax(decoded_nodes)[:n, 1]

    @staticmethod
    def prediction_loss(decoded_nodes, mask, candidates, energies, inv_temp: float):
        candidates = vmap_one_hot(candidates, 2)  # (B*N, K, 2))

        log_prob = vmap_compute_log_probs(
            decoded_nodes, mask, candidates
        )  # (B*N, K, 2)

        weights = jax.nn.softmax(-inv_temp * energies)  # (B*N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(mask)  # ()

        return loss

    @staticmethod
    def entropy_loss(decoded_nodes, mask):
        decoded_nodes = decoded_nodes * mask[:, None]
        prob = jax.nn.softmax(decoded_nodes)
        entropies = jnp.sum(jax.scipy.special.entr(prob), axis=1) / jnp.log(2)
        loss = -jnp.sum(jnp.log2(entropies), axis=0) / jnp.sum(mask)
        return loss

    @staticmethod
    def local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        if constraint_graph is None:
            raise ValueError("Constraint graph is None. Cannot calculate Lovasz loss.")
        log_probs = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
        n = jnp.shape(decoded_nodes)[0]
        # constraint_nodes = jnp.unique(constraint_graph.senders)
        # constraint_node_mask = utils.segment_sum(
        #    jnp.ones(len(constraint_nodes)), constraint_nodes, num_segments=n
        # )
        # constraint_node_mask = jnp.zeros(n)
        # for x in constraint_nodes:
        #    constraint_node_mask = constraint_node_mask.at[x].set(1)
        # constraint_node_mask = jnp.array(jnp.logical_not(mask), dtype=int)
        relevant_log_probs = jnp.sum(
            log_probs[graph.senders] * jnp.logical_not(graph.edges), axis=1
        )
        convolved_log_probs = utils.segment_sum(
            relevant_log_probs, graph.receivers, num_segments=n
        )

        lhs_values = convolved_log_probs  # * constraint_mask <- we do this later!
        constraint_senders = jnp.array(constraint_graph.senders, int)
        constraint_receivers = jnp.array(constraint_graph.receivers, int)

        relevant_log_x = log_probs[constraint_senders][:, 1]
        rhs_values = utils.segment_sum(
            data=relevant_log_x,
            segment_ids=constraint_receivers,
            num_segments=n,
        )
        rhs_values = (
            rhs_values + log_probs[:, 0]
        )  # * constraint_mask <- we do this later!
        difference = (jnp.exp(lhs_values) - jnp.exp(rhs_values)) * constraint_mask
        max_array = jnp.maximum(difference, jnp.zeros(len(rhs_values)))
        loss = jnp.sum(max_array) / jnp.sum(constraint_mask)
        return loss


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
    @abstractmethod
    def get_padded_candidate(solution_dict, n_nodes):
        #
        return np.pad(
            solution_dict,
            pad_width=(
                (0, 0),
                (0, int(np.ceil(n_nodes / 2)) - np.shape(solution_dict)[1]),
            ),  # @TODO: check if this is correct
        )

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
        sparse_clause_matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), (n, m)
        )
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
        senders, receivers = neighbors_list
        n_edge = len(senders)
        graph = jraph.GraphsTuple(
            n_node=np.asarray([m]),
            n_edge=np.asarray([n_edge]),
            senders=senders,
            receivers=receivers,
            globals=None,
            edges=np.zeros(n_edge),
            nodes=np.zeros(m),
        )
        return graph

    # @partial(jax.jit, static_argnames=("problem",))
    @staticmethod
    def get_violated_constraints(problem, assignment):
        def one_hot(x, k, dtype=jnp.float32):
            """Create a one-hot encoding of x of size k."""
            return jnp.array(x[:, None] == jnp.arange(k), dtype)

        graph = problem.graph
        n, m, _ = problem.params
        # this is required because we added edges to connect literal nodes
        receivers = graph.receivers[:-n]
        senders = graph.senders[:-n]
        new_assignment = jnp.ravel(one_hot(assignment, 2))
        edge_is_satisfied = jnp.ravel(
            new_assignment[None].T[senders].T
        )  # + np.ones(len(senders)), 2)
        number_of_literals_satisfied = utils.segment_sum(
            data=edge_is_satisfied, segment_ids=receivers, num_segments=2 * n + m
        )[2 * n :]
        return jnp.where(number_of_literals_satisfied > 0, 0, 1)

    @staticmethod
    def get_mask(n, n_node):
        return np.arange(n_node) < 2 * n

    @staticmethod
    def get_model_probabilities(decoded_nodes, n):
        if np.shape(decoded_nodes)[0] % 2 == 1:
            decoded_nodes = jnp.vstack((jnp.asarray(decoded_nodes), [[0]]))
            conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        else:
            conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        return jax.nn.softmax(conc_decoded_nodes)[:n, 1]

    @staticmethod
    def prediction_loss(decoded_nodes, mask, candidates, energies, inv_temp: float):
        # if np.shape(decoded_nodes)[0] % 2 == 1:
        #    conc_decoded_nodes = jnp.vstack((jnp.asarray(decoded_nodes), [0]))
        #    conc_decoded_nodes = jnp.reshape(conc_decoded_nodes, (-1, 2))
        #    new_mask = jnp.hstack((jnp.asarray(mask), [0]))
        #    new_mask = jnp.reshape(new_mask, (-1, 2))
        #    new_mask = new_mask[:, 0]
        # else:
        conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        new_mask = jnp.reshape(mask, (-1, 2))
        new_mask = new_mask[:, 0]
        decoded_nodes = conc_decoded_nodes * new_mask[:, None]
        candidates = vmap_one_hot(candidates, 2)
        # energies = energies[: len(new_mask), :]

        log_prob = vmap_compute_log_probs(
            decoded_nodes, new_mask, candidates
        )  # (B*N, K, 2)

        weights = jax.nn.softmax(-inv_temp * energies)  # (B*N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(
            mask
        )  # / 2  # ()
        return loss / 2

    @staticmethod
    def entropy_loss(decoded_nodes, mask):
        decoded_nodes = decoded_nodes * mask[:, None]
        # if np.shape(decoded_nodes)[0] % 2 == 1:
        #    decoded_nodes = jnp.vstack((jnp.asarray(decoded_nodes), [[0]]))
        #    conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        # else:
        conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        prob = jax.nn.softmax(conc_decoded_nodes)
        entropies = jnp.sum(jax.scipy.special.entr(prob), axis=1) / jnp.log(2)
        loss = -jnp.sum(jnp.log2(entropies), axis=0) / jnp.sum(mask)
        return loss

    @staticmethod
    def local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        if constraint_graph is None:
            raise ValueError("Constraint graph is None. Cannot calculate Lovasz loss.")
        n = jnp.shape(decoded_nodes)[0]
        # constraint_nodes = jnp.unique(constraint_graph.senders)
        # print(constraint_nodes)
        # constraint_node_mask = utils.segment_sum(
        #    jnp.ones(len(constraint_nodes)), constraint_nodes, num_segments=n
        # )
        # constraint_node_mask = jnp.array(jnp.logical_not(mask), dtype=int)
        # if jnp.shape(decoded_nodes)[0] % 2 == 1:
        #    new_decoded_nodes = jnp.vstack((jnp.asarray(decoded_nodes), [[0]]))
        #    new_decoded_nodes = jnp.reshape(new_decoded_nodes, (-1, 2))
        #    new_decoded_nodes = jnp.flip(new_decoded_nodes, axis=1)
        #    log_probs = jax.nn.log_softmax(new_decoded_nodes)
        #    log_probs = jnp.ravel(log_probs)[:-1]
        #
        # else:
        new_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        new_decoded_nodes = jnp.flip(new_decoded_nodes, axis=1)
        log_probs = jax.nn.log_softmax(new_decoded_nodes)
        log_probs = jnp.ravel(log_probs)
        masked_log_probs = log_probs * mask
        relevant_log_probs = masked_log_probs[graph.senders]
        convolved_log_probs = utils.segment_sum(
            relevant_log_probs, graph.receivers, num_segments=n
        )
        lhs_values = convolved_log_probs  # * constraint_mask
        # print(
        #    "lhs",
        #    np.exp(lhs_values) * constraint_mask,
        #    np.sum(np.exp(lhs_values) * constraint_mask),
        # )
        constraint_senders = jnp.array(constraint_graph.senders, int)
        constraint_receivers = jnp.array(constraint_graph.receivers, int)
        x_sigmoid = jnp.ravel(jax.nn.sigmoid(decoded_nodes))
        relevant_x_sigmoid = x_sigmoid[constraint_senders]
        rhs_values = utils.segment_sum(
            data=jnp.ravel(jnp.log(1 - relevant_x_sigmoid)),
            segment_ids=constraint_receivers,
            num_segments=n,
        )
        rhs_values = rhs_values + jnp.log(x_sigmoid)  # * constraint_mask
        # print(
        #    "rhs",
        #    np.exp(rhs_values) * constraint_mask,
        #    np.sum(np.exp(rhs_values) * constraint_mask),
        # )
        difference = (jnp.exp(lhs_values) - jnp.exp(rhs_values)) * constraint_mask
        max_array = jnp.maximum(difference, jnp.zeros(len(rhs_values)))
        loss = jnp.sum(max_array) / jnp.sum(constraint_mask)
        return loss


# Auxiliary functions ####


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


vmap_one_hot = jax.vmap(one_hot, in_axes=(0, None), out_axes=0)


def compute_log_probs(decoded_nodes, mask, candidate):
    a = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
    return candidate * a


vmap_compute_log_probs = jax.vmap(
    compute_log_probs, in_axes=(None, None, 1), out_axes=1
)
