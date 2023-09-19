"""file to test the loss functions."""
import sys
from unittest import TestCase
from allpairspy import AllPairs
import numpy as np
import pytest
from pysat.formula import CNF
from python.src.sat_instances import get_problem_from_cnf
from python.src.sat_representations import SATRepresentation, LCG, VCG
from python.src.data_utils import SATTrainingDataset, JraphDataLoader


sys.path.append("../../")


def one_hot(x_array, k_variable, dtype=np.float32):
    """Create a one-hot encoding of x of size k."""

    return np.array(x_array[:, None] == np.arange(k_variable), dtype)


def create_simple_neighbor_cnf(
    n_variables, n_clauses, k_locality, rep: SATRepresentation
):
    """Create a simple neighbor cnf formula with n variables, m clauses and locality k."""
    formula = CNF()
    for counter in range(n_clauses):
        clause = []
        for j in range(k_locality):
            clause.append(-1 * ((counter + j) % n_variables + 1))
        formula.append(clause)
    problem = get_problem_from_cnf(formula, rep)
    return problem


def get_decoded_nodes_on_solution_vcg(candidates):
    """Get decoded nodes on solution for VCG."""
    decoded_nodes = np.array(one_hot(candidates[:, 0], 2), dtype=float) * 100000
    return decoded_nodes


def get_decoded_nodes_on_solution_lcg(candidates):
    """Get decoded nodes on solution for LCG."""
    decoded_nodes = np.array(one_hot(candidates[:, 0], 2), dtype=float) * 100000
    decoded_nodes = np.ravel(decoded_nodes)
    return decoded_nodes


class LossTesting(TestCase):
    """Do loss testing."""

    def test_entropy_loss_uniform_decoded_nodes_vcg(self):
        "Test entropy loss for VCG. If all probabilities are 1/2, it should be zero."
        n_variables = 10
        n_clauses = 25
        decoded_nodes = np.ones((n_variables + n_clauses, 2)) / 2
        mask = VCG.get_mask(n_variables, n_variables + n_clauses)
        self.assertEqual(VCG.entropy_loss(decoded_nodes, mask), 0)

        n_variables = 9
        n_clauses = 26
        decoded_nodes = np.ones((n_variables + n_clauses, 2)) / 2
        mask = VCG.get_mask(n_variables, n_variables + n_clauses)
        self.assertEqual(VCG.entropy_loss(decoded_nodes, mask), 0)

    def test_entropy_loss_uniform_decoded_nodes_lcg(self):
        "Test entropy loss for LCG. If all probabilities are 1/2, it should be zero."
        n_variables = 10
        n_clauses = 25
        if n_clauses % 2 == 1:
            n_clauses += 1
        decoded_nodes = np.ones((2 * n_variables + n_clauses, 1)) / 2
        mask = LCG.get_mask(n_variables, 2 * n_variables + n_clauses)
        self.assertEqual(LCG.entropy_loss(decoded_nodes, mask), 0)

        n_variables = 9
        n_clauses = 26
        if n_clauses % 2 == 1:
            n_clauses += 1
        decoded_nodes = np.ones((2 * n_variables + n_clauses, 1)) / 2
        mask = LCG.get_mask(n_variables, 2 * n_variables + n_clauses)
        self.assertEqual(LCG.entropy_loss(decoded_nodes, mask), 0)

    def test_lll_loss(self):
        """Test LLL loss. If oracle outputs solution with probability one, loss should be zero."""
        n_variables = 10
        n_clauses = 26
        k_locality = 2

        # vcg
        problem = create_simple_neighbor_cnf(
            n_variables, n_clauses, k_locality, rep=VCG
        )
        graph = problem.graph
        decoded_nodes = np.vstack(
            [
                10000 * np.ones((n_variables + n_clauses)),
                np.zeros((n_variables + n_clauses)),
            ]
        ).T
        mask = VCG.get_mask(n_variables, n_variables + n_clauses)
        constraint_mask = np.array(np.logical_not(mask), dtype=int)
        neighbors_list = VCG.get_constraint_graph(
            n_variables, n_clauses, graph.senders, graph.receivers
        )
        loss = VCG.local_lovasz_loss(
            decoded_nodes, mask, graph, neighbors_list, constraint_mask
        )
        self.assertEqual(loss, 0)
        # lcg
        problem = create_simple_neighbor_cnf(
            n_variables, n_clauses, k_locality, rep=LCG
        )
        graph = problem.graph
        decoded_nodes = []
        for i in range(2 * n_variables + n_clauses):
            if i % 2 == 0:
                decoded_nodes.append(10000)
            else:
                decoded_nodes.append(0)
        decoded_nodes = np.array([decoded_nodes], dtype=float).T
        assert decoded_nodes.shape == (2 * n_variables + n_clauses, 1)
        mask = LCG.get_mask(n_variables, 2 * n_variables + n_clauses)
        constraint_mask = np.array(np.logical_not(mask), dtype=int)
        neighbors_list = LCG.get_constraint_graph(
            n_variables, n_clauses, graph.senders, graph.receivers
        )
        loss = LCG.local_lovasz_loss(
            decoded_nodes, mask, graph, neighbors_list, constraint_mask
        )
        self.assertEqual(loss, 0)


pairs = list(
    AllPairs(
        [
            [
                "python/tests/test_instances/single_instance/",
                "python/tests/test_instances/multiple_instances/",
            ],
            [VCG, LCG],
            [1, 2],
            [True, False],
        ]
    )
)


class TestParameterized:
    """Do the test."""

    @pytest.mark.parametrize(
        ["data_dir", "representation", "batch_size", "return_candidates"],
        pairs,
    )
    def test_lll_on_solution(
        self,
        data_dir,
        representation,
        batch_size,
        return_candidates,
    ):
        """Test LLL loss on solution"""
        sat_data = SATTrainingDataset(
            data_dir,
            representation=representation,
            return_candidates=return_candidates,
            include_constraint_graph=True,
        )
        data_loader = JraphDataLoader(sat_data, batch_size=batch_size, shuffle=False)

        for _, batch in enumerate(data_loader):
            (mask, graph, neighbors_list, constraint_mask), (
                candidates,
                _,
            ) = batch
            if representation == VCG:
                decoded_nodes = get_decoded_nodes_on_solution_vcg(candidates)

            if representation == LCG:
                decoded_nodes = get_decoded_nodes_on_solution_lcg(candidates)

            loss = representation.local_lovasz_loss(
                decoded_nodes, mask, graph, neighbors_list, constraint_mask
            )
            assert loss == 0

    @pytest.mark.parametrize(
        ["data_dir", "representation", "batch_size", "return_candidates"],
        pairs,
    )
    def test_gibbs_on_solution(
        self,
        data_dir,
        representation,
        batch_size,
        return_candidates,
    ):
        """Test gibbs loss on solution. It should give zero if oracle samples solution with probability one."""
        sat_data = SATTrainingDataset(
            data_dir,
            representation=representation,
            return_candidates=return_candidates,
            include_constraint_graph=False,
        )
        data_loader = JraphDataLoader(sat_data, batch_size=batch_size, shuffle=False)
        for _, batch in enumerate(data_loader):
            (mask, _, _, _), (
                candidates,
                energies,
            ) = batch
            if representation == VCG:
                decoded_nodes = get_decoded_nodes_on_solution_vcg(candidates)

            if representation == LCG:
                decoded_nodes = get_decoded_nodes_on_solution_lcg(candidates)

            inv_temp = 1e10
            loss = representation.prediction_loss(
                decoded_nodes, mask, candidates, energies, inv_temp=inv_temp
            )
            assert loss == 0
