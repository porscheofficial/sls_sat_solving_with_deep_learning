"""Definition of SATRepresentations and the corresponding helper functions. Namely, this is done for LCG and VCG representation."""
from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import jax
from jraph._src import utils
from pysat.formula import CNF
import scipy
import jraph
from scipy.sparse._sparsetools import expandptr


class SATRepresentation(ABC):
    """Definition of SATRepresentation-class with certain helper functions for the specified graph definition.

    Args:
        ABC (_type_): @TODO: _description_
    """

    @staticmethod
    @abstractmethod
    def get_graph(n_variables, n_clauses, clauses, clause_lengths):
        """Return the graph items: nodes, senders, receivers, edges, n_node, n_edge for input n = number of variables, m = number of clauses, clauses = clauses to embedd in the graph, clause_length = containing length of clauses.

        Args:
            n_variables (int): number of variables
            n_clauses (int): number of clauses
            clauses (array): clauses of the formula in CNF-form
            clause_lengths (array / list): array containing the length of clauses
        """
        # pass

    @staticmethod
    @abstractmethod
    def get_constraint_graph(
        n_variables, n_clauses, senders, receivers
    ) -> jraph.GraphsTuple:
        """Compute the constraint graph from n = number of variables, m = number of clauses and senders, receivers from the graph (encoding variable occurence in clauses). This constraint graph has edges between clauses whenever they share variables. This is used for the Lovasz Local Lemma Loss.

        Args:
            n_variables (int): number of variables
            n_clauses (int): number of clauses
            senders (array): sending nodes (variable nodes)
            receivers (array): receiving nodes (clause nodes)

        Returns:
            jraph.GraphsTuple: constraint graph encoding the neighborhood of clauses
        """
        # pass

    @staticmethod
    @abstractmethod
    def get_violated_constraints(problem, assignment):
        """Input a problem and an assignment, return whether clauses are violated. Returns an array which has a "1" for each clause that is violated and a "0" for a non-violated clause.

        Args:
            problem (@TODO: type): problem input
            assignment (list / array): assignment input we want to check for violated constraints

        Returns:
            jnp.array: array encoding whether assignment violates the clauses ("1" for clause is violated and "0" for clause is not violated by current assignment).
        """
        # pass

    @staticmethod
    @abstractmethod
    def get_mask(n_variables, n_node):
        """Return mask encoding which nodes are variable nodes ("1" for nodes that are variable nodes and "0" for nodes that are not variable nodes (i.e. padding nodes or constraint nodes)).

        Args:
            n_variables (int): number of variables
            n_node (int): number of nodes in the graph

        Returns:
            array: mask array that encodes which nodes are variable nodes and which are not.
        """
        # pass

    @staticmethod
    @abstractmethod
    def get_n_nodes(cnf: CNF):
        """Return number of nodes for the input cnf formula.

        Args:
            cnf (CNF): input cnf formula

        Returns:
            int: number of nodes in the graph
        """
        # pass

    @staticmethod
    @abstractmethod
    def get_n_edges(cnf: CNF):
        """Return number of edges.

        Args:
            cnf (CNF): input cnf formula

        Returns:
            int: number of edges in the graph
        """
        # pass

    @staticmethod
    @abstractmethod
    def get_padded_candidate(solution_dict, n_nodes):
        """Pad the candidates: we pad the solution (containing n = number of variables elements on one axis and c = number of candidates on the other axis) and pad it on the number of nodes n_nodes in the graph with zeros.

        Args:
            solution_dict (list/array): solution string of the given graph
            n_nodes (int): number of nodes in the graph

        Returns:
            array: returns padded solution / candidates
        """
        # pass

    @staticmethod
    @abstractmethod
    def get_model_probabilities(decoded_nodes, n_variables):
        """Return, for each, problem variable, the Bernoulli parameter of the model for this variable.

        That is, the ith value of the returned array is the probability with which the model will assign 1 to the
        ith variable. The reasoning for choosing the first, rather than the zeroth, column of the model output below is as follows: When evaluating the loss function, candidates are one-hot encoded, which means that when a satisfying assignment
        for a problem sets variable i to 1, then this will increase the likelihood that the model will set this variable to
        1, meaning, all else being equal, a larger Bernoulli weight in element [i,1] of the model output. As a result the
        right column of the softmax of the model output equals the models likelihood for setting variables to 1, which is
        what we seek.

        Args:
            decoded_nodes (array): decoded nodes output of the model which is two-dimensional for VCG.
            n_variables (int): number of variables

        Returns:
            array: probability vector encoding probability for every variable to sample a one according to the Neural Network oracle.
        """
        # pass

    @staticmethod
    @abstractmethod
    def prediction_loss(decoded_nodes, mask, candidates, energies, inv_temp: float):
        """Return loss term inspired by https://arxiv.org/abs/2012.13349. Returns the Gibbs-Loss.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            candidates (array): array containing the solution and all candidates
            energies (array): array containing the number of clauses violated by the candidates in candidates-array from above
            inv_temp (float): inverse temperature parameter used in the loss function / Gibbs formula

        Returns:
            float: corresponding loss value for the input
        """
        # pass

    @staticmethod
    @abstractmethod
    def local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        """Return Local Lovasz Lemma term inspired by the famous Local Lovasz Lemma (see https://arxiv.org/abs/0903.0544) -> expression zero means that Moser's algorithm will return solution efficiently.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the graph for the problem at hand
            constraint_graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the constraint-graph for the problem at hand (edges whenever constraints share variables)
            constraint_mask (array): mask ecoding which nodes are constraint nodes (contains a "1" for them and a "0" for all other nodes)

        Raises:
            ValueError: if we have no constraint graph calculated, we raise an error since it is needed for this loss term.

        Returns:
            float: corresponding loss value for the input
        """
        # pass

    @staticmethod
    @abstractmethod
    def alt_local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        """Return Local Lovasz Lemma term inspired by the famous Local Lovasz Lemma (see https://arxiv.org/abs/0903.0544) -> expression zero means that Moser's algorithm will return solution efficiently. We use here a slightly modiefied version of the Lemma. Please have a look at our paper for the details.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the graph for the problem at hand
            constraint_graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the constraint-graph for the problem at hand (edges whenever constraints share variables)
            constraint_mask (array): mask ecoding which nodes are constraint nodes (contains a "1" for them and a "0" for all other nodes)

        Raises:
            ValueError: if we have no constraint graph calculated, we raise an error since it is needed for this loss term.

        Returns:
            float: corresponding loss value for the input
        """
        # pass

    @staticmethod
    @abstractmethod
    def entropy_loss(decoded_nodes, mask):
        """Return entropy loss term making sure that oracle does not get too deterministic.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)

        Returns:
            float: corresponding loss value for the input
        """
        # pass


class VCG(SATRepresentation):
    """class for VCG encoding."""

    @staticmethod
    @abstractmethod
    def get_n_nodes(cnf: CNF):
        """Return number of nodes for the input cnf formula.

        Args:
            cnf (CNF): input cnf formula

        Returns:
            int: number of nodes in the graph in VCG representation = n_claues+n_variables
        """
        n_variables = cnf.nv
        n_clauses = len(cnf.clauses)
        return n_variables + n_clauses

    @staticmethod
    @abstractmethod
    def get_n_edges(cnf: CNF):
        """Return number of edges.

        Args:
            cnf (CNF): input cnf formula

        Returns:
            int: number of edges in the graph in VCG representation
        """
        return sum([len(clause) for clause in cnf.clauses])

    @staticmethod
    @abstractmethod
    def get_padded_candidate(solution_dict, n_nodes):
        """Pad the candidates here: we pad the solution (containing n = number of variables elements on one axis and c = number of candidates on the other axis) and pad it on the number of nodes n_nodes in the graph with zeros.

        Args:
            solution_dict (list/array): solution string of the given graph
            n_nodes (int): number of nodes in the graph

        Returns:
            array: returns padded solution / candidates
        """
        return np.pad(
            solution_dict,
            pad_width=(
                (0, 0),
                (0, int(np.ceil(n_nodes)) - np.shape(solution_dict)[1]),
            ),
        )

    @staticmethod
    def get_graph(n_variables, n_clauses, clauses, clause_lengths):
        """Return the graph items: nodes, senders, receivers, edges, n_node, n_edge for input n = number of variables, m = number of clauses, clauses = clauses to embedd in the graph, clause_length = containing length of clauses.

        Args:
            n_variables (int): number of variables
            n_clauses (int): number of clauses
            clauses (array): clauses of the formula in CNF-form
            clause_lengths (array / list): array containing the length of clauses

        Returns:
            nodes (array): assigned node values
            senders (array): sending nodes (variable nodes)
            receivers (array): receiving nodes (clause nodes)
            edges (array): assigned edge values
            n_node (int): number of nodes in the graph
            n_edge (int): number of edges in the graph
        """
        n_node = n_variables + n_clauses
        n_edge = sum(clause_lengths)

        edges = []
        senders = []
        receivers = []
        nodes = [0 if i < n_variables else 1 for i in range(n_node)]
        for j, clause in enumerate(clauses):
            support = [(abs(literal) - 1) for literal in clause]
            assert len(support) == len(
                set(support)
            ), "Multiple occurrences of single variable in constraint"

            vals = ((np.sign(clause) + 1) // 2).astype(np.int32)

            senders.extend(support)
            edges.extend(vals)
            receivers.extend(np.repeat(j + n_variables, len(clause)))

        return nodes, senders, receivers, edges, n_node, n_edge

    @staticmethod
    def get_constraint_graph(
        n_variables, n_clauses, senders, receivers
    ) -> jraph.GraphsTuple:
        """Compute the constraint graph from n = number of variables, m = number of clauses and senders, receivers from the graph (encoding variable occurence in clauses). This constraint graph has edges between clauses whenever they share variables. This is used for the Lovasz Local Lemma Loss.

        Args:
            n_variables (int): number of variables
            n_clauses (int): number of clauses
            senders (array): sending nodes (variable nodes)
            receivers (array): receiving nodes (clause nodes)

        Returns:
            jraph.GraphsTuple: constraint graph encoding the neighborhood of clauses
        """
        row_ind = np.asarray(senders)
        col_ind = np.asarray(receivers) - n_variables * np.ones(len(receivers))
        data = np.ones(len(row_ind))
        sparse_clause_matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), (n_variables, n_clauses)
        )
        adj_matrix = sparse_clause_matrix.transpose() @ sparse_clause_matrix
        major_dimension, _ = adj_matrix.shape
        minor_indices = adj_matrix.indices
        major_indices = np.empty(len(minor_indices), dtype=adj_matrix.indices.dtype)
        expandptr(major_dimension, adj_matrix.indptr, major_indices)
        dummy_x, dummy_y = np.array(
            np.where(
                minor_indices - major_indices != 0,
                [minor_indices + n_variables, major_indices + n_variables],
                0,
            )
        )
        dummy_x = dummy_x[dummy_x != 0]
        dummy_y = dummy_y[dummy_y != 0]
        neighbors_list = np.vstack((dummy_y, dummy_x))
        graph = jraph.GraphsTuple(
            n_node=np.asarray([n_clauses]),
            n_edge=np.asarray([len(dummy_x)]),
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
        """Input a problem and an assignment, return whether clauses are violated. Returns an array which has a "1" for each clause that is violated and a "0" for a non-violated clause.

        Args:
            problem (@TODO: type): problem input
            assignment (list / array): assignment input we want to check for violated constraints

        Returns:
            jnp.array: array encoding whether assignment violates the clauses ("1" for clause is violated and "0" for clause is not violated by current assignment).
        """
        graph = problem.graph
        n_variables, n_clauses, _ = problem.params
        edge_is_satisfied = graph.edges[:, 1] == assignment.T[graph.senders].T
        number_of_literals_satisfied = utils.segment_sum(
            data=edge_is_satisfied.astype(int),
            segment_ids=graph.receivers - n_variables,
            num_segments=n_clauses,
        )
        return jnp.where(number_of_literals_satisfied > 0, 0, 1)

    @staticmethod
    def get_mask(n_variables, n_node):
        """Return mask encoding which nodes are variable nodes ("1" for nodes that are variable nodes and "0" for nodes that are not variable nodes (i.e. padding nodes or constraint nodes)).

        Args:
            n_variables (int): number of variables
            n_node (int): number of nodes in the graph

        Returns:
            array: mask array that encodes which nodes are variable nodes and which are not.
        """
        return np.arange(n_node) < n_variables

    @staticmethod
    def get_model_probabilities(decoded_nodes, n_variables):
        """Return, for each, problem variable, the Bernoulli parameter of the model for this variable.

        That is, the ith value of the returned array is the probability with which the model will assign 1 to the
        ith variable. The reasoning for choosing the first, rather than the zeroth, column of the model output below is as follows:
        When evaluating the loss function, candidates are one-hot encoded, which means that when a satisfying assignment
        for a problem sets variable i to 1, then this will increase the likelihood that the model will set this variable to
        1, meaning, all else being equal, a larger Bernoulli weight in element [i,1] of the model output. As a result the
        right column of the softmax of the model output equals the models likelihood for setting variables to 1, which is
        what we seek.

        Args:
            decoded_nodes (array): decoded nodes output of the model which is two-dimensional for VCG.
            n_variables (int): number of variables

        Returns:
            array: probability vector encoding probability for every variable to sample a one according to the Neural Network oracle.
        """
        return jax.nn.softmax(decoded_nodes)[:n_variables, 1]

    @staticmethod
    def prediction_loss(decoded_nodes, mask, candidates, energies, inv_temp: float):
        """Return Loss term inspired by https://arxiv.org/abs/2012.13349 -> Returns the Gibbs-Loss.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            candidates (array): array containing the solution and all candidates
            energies (array): array containing the number of clauses violated by the candidates in candidates-array from above
            inv_temp (float): inverse temperature parameter used in the loss function / Gibbs formula

        Returns:
            float: corresponding loss value for the input
        """
        candidates = vmap_one_hot(candidates, 2)  # (B*N, K, 2))

        log_prob = vmap_compute_log_probs(
            decoded_nodes, mask, candidates
        )  # (B*N, K, 2)

        weights = jax.nn.softmax(-inv_temp * energies)  # (B*N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(mask)  # ()

        return loss

    @staticmethod
    def entropy_loss(decoded_nodes, mask):
        """Return Entropy loss term making sure that oracle does not get too deterministic.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)

        Returns:
            float: corresponding loss value for the input
        """
        decoded_nodes = decoded_nodes * mask[:, None]
        prob = jax.nn.softmax(decoded_nodes)
        entropies = jnp.sum(jax.scipy.special.entr(prob), axis=1) / jnp.log(2)
        loss = -jnp.sum(jnp.log2(entropies), axis=0) / jnp.sum(mask)
        return loss

    @staticmethod
    def local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        """Return Local Lovasz Lemma term inspired by the famous Local Lovasz Lemma (see https://arxiv.org/abs/0903.0544)-> expression zero means that Moser's algorithm will return solution efficiently.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the graph for the problem at hand
            constraint_graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the constraint-graph for the problem at hand (edges whenever constraints share variables)
            constraint_mask (array): mask ecoding which nodes are constraint nodes (contains a "1" for them and a "0" for all other nodes)

        Raises:
            ValueError: if we have no constraint graph calculated, we raise an error since it is needed for this loss term.

        Returns:
            float: corresponding loss value for the input
        """
        if constraint_graph is None:
            raise ValueError("Constraint graph is None. Cannot calculate Lovasz loss.")
        log_probs = jax.nn.log_softmax(decoded_nodes)
        masked_log_probs = log_probs * mask[:, None]
        n_variables = jnp.shape(decoded_nodes)[0]
        relevant_log_probs = jnp.sum(
            masked_log_probs[graph.senders] * jnp.logical_not(graph.edges), axis=1
        )
        convolved_log_probs = utils.segment_sum(
            relevant_log_probs, graph.receivers, num_segments=n_variables
        )

        lhs_values = convolved_log_probs
        constraint_senders = jnp.array(constraint_graph.senders, int)
        constraint_receivers = jnp.array(constraint_graph.receivers, int)

        relevant_log_x = log_probs[constraint_senders][:, 1]
        rhs_values = utils.segment_sum(
            data=relevant_log_x,
            segment_ids=constraint_receivers,
            num_segments=n_variables,
        )
        rhs_values = rhs_values + log_probs[:, 0]
        difference = (jnp.exp(lhs_values) - jnp.exp(rhs_values)) * constraint_mask
        max_array = jnp.maximum(difference, jnp.zeros(len(rhs_values)))
        loss = jnp.linalg.norm(max_array, 2)
        return loss

    @staticmethod
    def alt_local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        """Return Local Lovasz Lemma term inspired by the famous Local Lovasz Lemma (see https://arxiv.org/abs/0903.0544) -> expression zero means that Moser's algorithm will return solution efficiently. We use here a slightly modiefied version of the Lemma. Please have a look at our paper for the details.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the graph for the problem at hand
            constraint_graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the constraint-graph for the problem at hand (edges whenever constraints share variables)
            constraint_mask (array): mask ecoding which nodes are constraint nodes (contains a "1" for them and a "0" for all other nodes)

        Raises:
            ValueError: if we have no constraint graph calculated, we raise an error since it is needed for this loss term.

        Returns:
            float: corresponding loss value for the input
        """
        if constraint_graph is None:
            raise ValueError("Constraint graph is None. Cannot calculate Lovasz loss.")
        log_probs = jax.nn.log_softmax(decoded_nodes)
        masked_log_probs = log_probs * mask[:, None]
        n_variables = jnp.shape(decoded_nodes)[0]
        relevant_log_probs = jnp.sum(
            masked_log_probs[graph.senders] * jnp.logical_not(graph.edges), axis=1
        )
        convolved_log_probs = utils.segment_sum(
            relevant_log_probs, graph.receivers, num_segments=n_variables
        )

        constraint_senders = jnp.array(constraint_graph.senders, int)
        constraint_receivers = jnp.array(constraint_graph.receivers, int)

        relevant_log_x = log_probs[constraint_senders][:, 1]
        log_neighborhood = utils.segment_sum(
            data=relevant_log_x,
            segment_ids=constraint_receivers,
            num_segments=n_variables,
        )
        log_neighborhood = log_neighborhood + log_probs[:, 1]

        lhs_values = (
            jnp.exp(convolved_log_probs - log_neighborhood - log_probs[:, 1])
        ) * constraint_mask
        rhs_values = jnp.exp((log_probs[:, 0] - log_probs[:, 1])) * constraint_mask

        difference = lhs_values - rhs_values
        max_array = jnp.maximum(difference, jnp.zeros(len(rhs_values)))
        loss = jnp.linalg.norm(max_array, 2)

        return loss


class LCG(SATRepresentation):
    """class of LCG representation."""

    @staticmethod
    @abstractmethod
    def get_n_nodes(cnf: CNF):
        """Return number of nodes for the input cnf formula.

        Args:
            cnf (CNF): input cnf formula

        Returns:
            int: number of nodes in the graph in LCG representation = 2*n+m
        """
        n_variables = cnf.nv
        n_clauses = len(cnf.clauses)
        return 2 * n_variables + n_clauses

    @staticmethod
    @abstractmethod
    def get_n_edges(cnf: CNF):
        """Return number of edges.

        In LCG, there is one edge for each literal in each clause, plus one
        edge for each variable that connects the positive and negative literal
        node.

        Args:
            cnf (CNF): input cnf formula

        Returns:
            int: number of edges in the graph in LCG representation
        """
        return sum([len(clause) for clause in cnf.clauses]) + cnf.nv

    @staticmethod
    @abstractmethod
    def get_padded_candidate(solution_dict, n_nodes):
        """Pad the candidates here: we pad the solution (containing n = number of variables elements on one axis and c = number of candidates on the other axis) and pad it on the number of nodes n_nodes in the graph with zeros.

        Args:
            solution_dict (list/array): solution string of the given graph
            n_nodes (int): number of nodes in the graph

        Returns:
            array: returns padded solution / candidates
        """
        return np.pad(
            solution_dict,
            pad_width=(
                (0, 0),
                (0, int(np.ceil(n_nodes / 2)) - np.shape(solution_dict)[1]),
            ),
        )

    @staticmethod
    def get_graph(n_variables, n_clauses, clauses, clause_lengths):
        """Return the graph items: nodes, senders, receivers, edges, n_node, n_edge for input n = number of variables, m = number of clauses, clauses = clauses to embedd in the graph, clause_length = containing length of clauses.

        Args:
            n_variables (int): number of variables
            n_clauses (int): number of clauses
            clauses (array): clauses of the formula in CNF-form
            clause_lengths (array / list): array containing the length of clauses

        Returns:
            nodes (array): assigned node values
            senders (array): sending nodes (variable nodes)
            receivers (array): receiving nodes (clause nodes)
            edges (array): assigned edge values
            n_node (int): number of nodes in the graph
            n_edge (int): number of edges in the graph
        """
        n_node = 2 * n_variables + n_clauses
        n_edge = sum(clause_lengths) + n_variables

        edges = []
        senders = []
        receivers = []

        # 1 indicates a literal node.
        # -1 indicated a negated literal node.
        # 0 indicates a constraint node.

        nodes = []
        for i in range(n_node):
            if i < 2 * n_variables:
                if i % 2 == 0:
                    nodes.append(1)
                if i % 2 == 1:
                    nodes.append(-1)
            else:
                nodes.append(0)
        for j, clause in enumerate(clauses):
            support = [(abs(literal) - 1) for literal in clause]
            assert len(support) == len(
                set(support)
            ), "Multiple occurrences of single variable in constraint"

            vals = ((np.sign(clause) + 1) // 2).astype(np.int32)

            for i, val in enumerate(vals):
                if val == 1:
                    senders.append(int(2 * support[i] + 1))
                else:
                    senders.append(int(2 * support[i]))
            edges.extend(np.repeat(0, len(clause)))
            receivers.extend(np.repeat(j + 2 * n_variables, len(clause)))

        for j in range(n_variables):
            senders.append(int(2 * j + 1))
            receivers.append(int(2 * j))
            edges.append(1)

        return nodes, senders, receivers, edges, n_node, n_edge

    @staticmethod
    def get_constraint_graph(n_variables, n_clauses, senders, receivers):
        """Compute the constraint graph from n = number of variables, m = number of clauses and senders, receivers from the graph (encoding variable occurence in clauses). This constraint graph has edges between clauses whenever they share variables. This is used for the Lovasz Local Lemma Loss.

        Args:
            n_variables (int): number of variables
            n_clauses (int): number of clauses
            senders (array): sending nodes (variable nodes)
            receivers (array): receiving nodes (clause nodes)

        Returns:
            jraph.GraphsTuple: constraint graph encoding the neighborhood of clauses
        """
        row_ind = np.floor(np.asarray(senders[:-n_variables]) / 2)
        col_ind = np.asarray(receivers[:-n_variables]) - 2 * n_variables * np.ones(
            len(receivers[:-n_variables])
        )
        data = np.ones(len(row_ind))
        sparse_clause_matrix = scipy.sparse.csr_matrix(
            (data, (row_ind, col_ind)), (n_variables, n_clauses)
        )
        adj_matrix = sparse_clause_matrix.transpose() @ sparse_clause_matrix
        major_dimension, _ = adj_matrix.shape
        minor_indices = adj_matrix.indices
        major_indices = np.empty(len(minor_indices), dtype=adj_matrix.indices.dtype)
        expandptr(major_dimension, adj_matrix.indptr, major_indices)
        dummy_x, dummy_y = np.array(
            np.where(
                minor_indices - major_indices != 0,
                [minor_indices + 2 * n_variables, major_indices + 2 * n_variables],
                0,
            )
        )
        dummy_x = dummy_x[dummy_x != 0]
        dummy_y = dummy_y[dummy_y != 0]
        neighbors_list = np.vstack((dummy_y, dummy_x))
        senders, receivers = neighbors_list
        n_edge = len(senders)
        graph = jraph.GraphsTuple(
            n_node=np.asarray([n_clauses]),
            n_edge=np.asarray([n_edge]),
            senders=senders,
            receivers=receivers,
            globals=None,
            edges=np.zeros(n_edge),
            nodes=np.zeros(n_clauses),
        )
        return graph

    # @partial(jax.jit, static_argnames=("problem",))
    @staticmethod
    def get_violated_constraints(problem, assignment):
        """Input a problem and an assignment, return whether clauses are violated. Returns an array which has a "1" for each clause that is violated and a "0" for a non-violated clause.

        Args:
            problem (SATProblem): problem input
            assignment (list / array): assignment input we want to check for violated constraints

        Returns:
            jnp.array: array encoding whether assignment violates the clauses ("1" for clause is violated and "0" for clause is not violated by current assignment).
        """
        graph = problem.graph
        n_variables, n_clauses, _ = problem.params
        # this is required because we added edges to connect literal nodes
        receivers = graph.receivers[:-n_variables]
        senders = graph.senders[:-n_variables]
        new_assignment = jnp.ravel(one_hot(assignment, 2))
        edge_is_satisfied = jnp.ravel(new_assignment[None].T[senders].T)
        number_of_literals_satisfied = utils.segment_sum(
            data=edge_is_satisfied,
            segment_ids=receivers,
            num_segments=2 * n_variables + n_clauses,
        )[2 * n_variables :]
        return jnp.where(number_of_literals_satisfied > 0, 0, 1)

    @staticmethod
    def get_mask(n_variables, n_node):
        """Return mask encoding which nodes are variable nodes ("1" for nodes that are variable nodes and "0" for nodes that are not variable nodes (i.e. padding nodes or constraint nodes)).

        Args:
            n_variables (int): number of variables
            n_node (int): number of nodes in the graph

        Returns:
            array: mask array that encodes which nodes are variable nodes and which are not.
        """
        return np.arange(n_node) < 2 * n_variables

    @staticmethod
    def get_model_probabilities(decoded_nodes, n_variables):
        """Return, for each, problem variable, the Bernoulli parameter of the model for this variable.

        That is, the ith value of the returned array is the probability with which the model will assign 1 to the
        ith variable. The reasoning for choosing the first, rather than the zeroth, column of the model output below is as follows:
        When evaluating the loss function, candidates are one-hot encoded, which means that when a satisfying assignment
        for a problem sets variable i to 1, then this will increase the likelihood that the model will set this variable to
        1, meaning, all else being equal, a larger Bernoulli weight in element [i,1] of the model output. As a result the
        right column of the softmax of the model output equals the models likelihood for setting variables to 1, which is
        what we seek. In LCG this is a bit more complicated for LCG compared to VCG since every literal has one node and we first need to concatenate the output for the positive and negative node to take the softmax then.

        Args:
            decoded_nodes (array): decoded nodes output of the model which is one-dimensional for LCG.
            n_variables (int): number of variables

        Returns:
            array: probability vector encoding probability for every variable to sample a one according to the Neural Network oracle.
        """
        if np.shape(decoded_nodes)[0] % 2 == 1:
            decoded_nodes = jnp.vstack((jnp.asarray(decoded_nodes), [[0]]))
            conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        else:
            conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        return jax.nn.softmax(conc_decoded_nodes)[:n_variables, 1]

    @staticmethod
    def prediction_loss(decoded_nodes, mask, candidates, energies, inv_temp: float):
        """Return Loss term inspired by https://arxiv.org/abs/2012.13349 -> Returns the Gibbs-Loss.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            candidates (array): array containing the solution and all candidates
            energies (array): array containing the number of clauses violated by the candidates in candidates-array from above
            inv_temp (float): inverse temperature parameter used in the loss function / Gibbs formula

        Returns:
            float: corresponding loss value for the input
        """
        conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        new_mask = jnp.reshape(mask, (-1, 2))
        new_mask = new_mask[:, 0]
        decoded_nodes = conc_decoded_nodes * new_mask[:, None]
        candidates = vmap_one_hot(candidates, 2)

        log_prob = vmap_compute_log_probs(
            decoded_nodes, new_mask, candidates
        )  # (B*N, K, 2)

        weights = jax.nn.softmax(-inv_temp * energies)  # (B*N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(mask)
        return loss / 2

    @staticmethod
    def entropy_loss(decoded_nodes, mask):
        """Return Entropy loss term making sure that oracle does not get too deterministic.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)

        Returns:
            float: corresponding loss value for the input
        """
        decoded_nodes = decoded_nodes * mask[:, None]
        conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        prob = jax.nn.softmax(conc_decoded_nodes)
        entropies = jnp.sum(jax.scipy.special.entr(prob), axis=1) / jnp.log(2)
        loss = -jnp.sum(jnp.log2(entropies), axis=0) / jnp.sum(mask)
        return loss

    @staticmethod
    def local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        """Return Local Lovasz Lemma term inspired by the famous Local Lovasz Lemma (see https://arxiv.org/abs/0903.0544) -> expression zero means that Moser's algorithm will return solution efficiently. Again, it is a bit more complicated for the LCG case since each literal has a node and to get the probability (or the log of it to be precise) for the variable nodes. We concatenate those (positive and negative literal) and then take the softmax. We take the sigmoid of decoded nodes of the constraint nodes for the x(c_i) assignment used for the Lovasz Local Lemma Theorem.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the graph for the problem at hand
            constraint_graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the constraint-graph for the problem at hand (edges whenever constraints share variables)
            constraint_mask (array): mask ecoding which nodes are constraint nodes (contains a "1" for them and a "0" for all other nodes)

        Raises:
            ValueError: if we have no constraint graph calculated, we raise an error since it is needed for this loss term.

        Returns:
            float: corresponding loss value for the input
        """
        if constraint_graph is None:
            raise ValueError("Constraint graph is None. Cannot calculate Lovasz loss.")
        n_variables = jnp.shape(decoded_nodes)[0]

        new_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        new_decoded_nodes = jnp.flip(new_decoded_nodes, axis=1)
        log_probs = jax.nn.log_softmax(new_decoded_nodes)
        log_probs = jnp.ravel(log_probs)
        masked_log_probs = log_probs * mask
        relevant_log_probs = masked_log_probs[graph.senders]
        convolved_log_probs = utils.segment_sum(
            relevant_log_probs, graph.receivers, num_segments=n_variables
        )
        lhs_values = convolved_log_probs
        constraint_senders = jnp.array(constraint_graph.senders, int)
        constraint_receivers = jnp.array(constraint_graph.receivers, int)
        x_sigmoid = jnp.ravel(jax.nn.sigmoid(decoded_nodes))
        relevant_x_sigmoid = x_sigmoid[constraint_senders]
        rhs_values = utils.segment_sum(
            data=jnp.ravel(jnp.log(1 - relevant_x_sigmoid)),
            segment_ids=constraint_receivers,
            num_segments=n_variables,
        )
        rhs_values = rhs_values + jnp.log(x_sigmoid)
        difference = (jnp.exp(lhs_values) - jnp.exp(rhs_values)) * constraint_mask
        max_array = jnp.maximum(difference, jnp.zeros(len(rhs_values)))

        loss = jnp.linalg.norm(max_array, 2)

        return loss

    @staticmethod
    def alt_local_lovasz_loss(
        decoded_nodes, mask, graph, constraint_graph, constraint_mask
    ):
        """Return Local Lovasz Lemma term inspired by the famous Local Lovasz Lemma (see https://arxiv.org/abs/0903.0544) -> expression zero means that Moser's algorithm will return solution efficiently. We use here a slightly modiefied version of the Lemma. Please have a look at our paper for the details.

        Args:
            decoded_nodes (array): GNN output for all the nodes
            mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
            graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the graph for the problem at hand
            constraint_graph (jraph.GraphsTuple): jraph.GraphsTuple encoding the constraint-graph for the problem at hand (edges whenever constraints share variables)
            constraint_mask (array): mask ecoding which nodes are constraint nodes (contains a "1" for them and a "0" for all other nodes)

        Raises:
            ValueError: if we have no constraint graph calculated, we raise an error since it is needed for this loss term.

        Returns:
            float: corresponding loss value for the input
        """
        if constraint_graph is None:
            raise ValueError("Constraint graph is None. Cannot calculate Lovasz loss.")
        n_variables = jnp.shape(decoded_nodes)[0]
        new_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
        new_decoded_nodes = jnp.flip(new_decoded_nodes, axis=1)
        log_probs = jax.nn.log_softmax(new_decoded_nodes)
        log_probs = jnp.ravel(log_probs)
        masked_log_probs = log_probs * mask
        relevant_log_probs = masked_log_probs[graph.senders]
        convolved_log_probs = utils.segment_sum(
            relevant_log_probs, graph.receivers, num_segments=n_variables
        )

        constraint_senders = jnp.array(constraint_graph.senders, int)
        constraint_receivers = jnp.array(constraint_graph.receivers, int)
        x_sigmoid = jnp.ravel(jax.nn.sigmoid(decoded_nodes))
        relevant_x_sigmoid = x_sigmoid[constraint_senders]
        prod_inclusive_neighborhood_values = utils.segment_sum(
            data=jnp.ravel(jnp.log(1 - relevant_x_sigmoid)),
            segment_ids=constraint_receivers,
            num_segments=n_variables,
        )
        lhs_values = jnp.exp(
            convolved_log_probs - prod_inclusive_neighborhood_values
        ) / (1 - x_sigmoid)
        rhs_values = x_sigmoid / (1 - x_sigmoid)

        difference = (lhs_values - rhs_values) * constraint_mask
        max_array = jnp.maximum(difference, jnp.zeros(len(difference)))
        loss = jnp.linalg.norm(max_array, 2)

        return loss


# Auxiliary functions ####


def one_hot(x_array, k_size, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k.

    Args:
        x_array (array): input that we want to one-hot encode
        k_size (int): one hot encoding of size k
        dtype (_type_, optional): type of the one-hot output. Defaults to jnp.float32.

    Returns:
        array: returns a one-hot encoded version of x of size k
    """
    return jnp.array(x_array[:, None] == jnp.arange(k_size), dtype)


vmap_one_hot = jax.vmap(one_hot, in_axes=(0, None), out_axes=0)


def compute_log_probs(decoded_nodes, mask, candidate):
    """Compute the logarithm of probabilities from decoded nodes.

    Args:
        decoded_nodes (array): GNN output for all the nodes
        mask (array): mask ecoding which nodes are variable nodes (contains a "1" for them and a "0" for all other nodes)
        candidate (array): array containing ONE candidate

    Returns:
        array: returns logarithm of probabilities
    """
    masked_softmax = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
    return candidate * masked_softmax


vmap_compute_log_probs = jax.vmap(
    compute_log_probs, in_axes=(None, None, 1), out_axes=1
)
