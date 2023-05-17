import numpy as np
import unittest
from python.src.sat_representations import SATRepresentation, LCG, VCG

from unittest import TestCase


class LossTesting(TestCase):
    def test_entropy_loss_uniform_decoded_nodes_vcg(self):
        n = 10
        m = 25
        decoded_nodes = np.ones((n + m, 2)) / 2
        mask = VCG.get_mask(n, n + m)
        self.assertEqual(VCG.entropy_loss(decoded_nodes, mask), 0)

    def test_entropy_loss_uniform_decoded_nodes_lcg(self):
        n = 10
        m = 25
        decoded_nodes = np.ones((2 * n + m, 1)) / 2
        mask = LCG.get_mask(n, 2 * n + m)
        self.assertEqual(LCG.entropy_loss(decoded_nodes, mask), 0)


"""
class TestLoss(unittest.TestCase):
    def entropy_loss_uniform_decoded_nodes_vcg(self):
        n = 10
        m = 25
        decoded_nodes = np.ones((n + m,2)) / 2
        mask = VCG.get_mask(n, n + m)
        self.assertEqual(VCG.entropy_loss(decoded_nodes, mask), 0)

    def entropy_loss_uniform_decoded_nodes_lcg(self):
        n = 10
        m = 25
        decoded_nodes = np.ones((2*n + m, 1)) / 2
        mask = LCG.get_mask(n, 2*n + m)
        self.assertEqual(LCG.entropy_loss(decoded_nodes, mask), 0)
    
    def entropy_loss_deterministic_decoded_nodes_lcg(self):
        n = 10
        m = 25
        decoded_nodes = np.vstack([np.ones((n + m)), np.zeros((n + m))]).T
        self.assertEqual(np.shape(decoded_nodes),(n + m, 2))
        #mask = VCG.get_mask(n, n + m)
        #self.assertEqual(LCG.entropy_loss(decoded_nodes, mask), 1)
"""
