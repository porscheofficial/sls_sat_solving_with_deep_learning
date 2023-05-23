import numpy as np
import unittest
from python.src.sat_instances import get_problem_from_cnf
from python.src.sat_representations import SATRepresentation, LCG, VCG
from pysat.formula import CNF
from unittest import TestCase


def create_simple_neighbor_cnf(n, m, k, rep: SATRepresentation):
    H = CNF()
    counter = 0
    for i in range(m):
        c = []
        for j in range(k):
            c.append(-1 * ((counter + j) % n + 1))
        counter += 1
        H.append(c)
    problem = get_problem_from_cnf(H, rep)
    return problem


class LossTesting(TestCase):
    def test_entropy_loss_uniform_decoded_nodes_vcg(self):
        n = 10
        m = 25
        decoded_nodes = np.ones((n + m, 2)) / 2
        mask = VCG.get_mask(n, n + m)
        self.assertEqual(VCG.entropy_loss(decoded_nodes, mask), 0)

        n = 9
        m = 26
        decoded_nodes = np.ones((n + m, 2)) / 2
        mask = VCG.get_mask(n, n + m)
        self.assertEqual(VCG.entropy_loss(decoded_nodes, mask), 0)

    def test_entropy_loss_uniform_decoded_nodes_lcg(self):
        n = 10
        m = 25
        if m % 2 == 1:
            m += 1
        decoded_nodes = np.ones((2 * n + m, 1)) / 2
        mask = LCG.get_mask(n, 2 * n + m)
        self.assertEqual(LCG.entropy_loss(decoded_nodes, mask), 0)

        n = 9
        m = 26
        if m % 2 == 1:
            m += 1
        decoded_nodes = np.ones((2 * n + m, 1)) / 2
        mask = LCG.get_mask(n, 2 * n + m)
        self.assertEqual(LCG.entropy_loss(decoded_nodes, mask), 0)

    # def test_neighbors_list(self):
    #    assert 0 == 0
    #    # tbd

    def test_LLL_loss(self):
        n = 10
        m = 26
        k = 2

        # vcg
        problem = create_simple_neighbor_cnf(n, m, k, rep=VCG)
        g = problem.graph
        decoded_nodes = np.vstack([10000 * np.ones((n + m)), np.zeros((n + m))]).T
        mask = VCG.get_mask(n, n + m)
        neighbors_list = VCG.get_constraint_graph(n, m, g.senders, g.receivers)
        loss = VCG.local_lovasz_loss(decoded_nodes, mask, g, neighbors_list)
        self.assertEqual(loss, 0)
        # lcg
        problem = create_simple_neighbor_cnf(n, m, k, rep=LCG)
        g = problem.graph
        decoded_nodes = []
        for i in range(2 * n + m):
            if i % 2 == 0:
                decoded_nodes.append(10000)
            else:
                decoded_nodes.append(0)
        decoded_nodes = np.array([decoded_nodes], dtype=float).T
        assert decoded_nodes.shape == (2 * n + m, 1)
        mask = LCG.get_mask(n, 2 * n + m)
        neighbors_list = LCG.get_constraint_graph(n, m, g.senders, g.receivers)
        loss = LCG.local_lovasz_loss(decoded_nodes, mask, g, neighbors_list)
        self.assertEqual(loss, 0)
