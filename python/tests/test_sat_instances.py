"""Test sat_instance functions."""
import sys
import unittest
from pysat.formula import CNF
import jax.numpy as jnp
import numpy as np
from python.src.sat_representations import VCG, LCG, SATRepresentation
from python.src.sat_instances import (
    all_bitstrings,
    get_problem_from_cnf,
)


sys.path.append("../../")


class TestConstraintProblemUtils(unittest.TestCase):
    """Test problem utility functions."""

    @staticmethod
    def instance_simple_tautology():
        """Define a simple tautology instance p cnf 1 1 \n 1 -1 0."""
        tautology = "p cnf 1 1 \n 1 -1 0"
        return CNF(from_string=tautology)

    @staticmethod
    def instance_simple_contradiction():
        """Define simple instance that leads to a contradiction p cnf 1 2 \n 1 0 \n -1 0."""
        contradiction = "p cnf 1 2 \n 1 0 \n -1 0"
        return CNF(from_string=contradiction)

    def test_forbid_multiple_variable_occurrences_in_single_clause(self):
        """Test whether multiple variable occurencies in a clause are detected."""
        try:
            get_problem_from_cnf(self.instance_simple_tautology(), representation=VCG)
        except AssertionError:
            pass
        self.assertRaises(AssertionError)

        try:
            get_problem_from_cnf(self.instance_simple_tautology(), representation=LCG)
        except AssertionError:
            pass
        self.assertRaises(AssertionError)

    def problem_parsing(self, rep: SATRepresentation):
        """Test problem parsing."""
        problem = get_problem_from_cnf(self.instance_simple_contradiction(), rep)
        self.assertEqual(problem.params, (1, 2, 1))

        string = "p cnf 3 2 \n 1 2 0 \n 3 0"
        cnf = CNF(from_string=string)
        problem = get_problem_from_cnf(cnf, rep)
        assert problem.params == (3, 2, 2)
        graph = problem.graph
        assert len(graph.nodes) == graph.n_node
        assert len(graph.receivers) == len(graph.senders)
        assert len(graph.senders) == len(graph.edges)
        assert len(graph.edges) == graph.n_edge

    def test_problem_parsing_lcg(self):
        """Test problem parsing with LCG."""
        self.problem_parsing(LCG)

    def test_problem_parsing_vcg(self):
        """Test problem parsing with VCG."""
        self.problem_parsing(VCG)

    def violated_constraints1(self, rep: SATRepresentation):
        """Test get_violated_constraints method (version 1)."""
        assignments = all_bitstrings(1)
        problem = get_problem_from_cnf(self.instance_simple_contradiction(), rep)
        self.assertEqual(
            [np.sum(rep.get_violated_constraints(problem, a)) for a in assignments],
            [1, 1],
        )

    def test_violated_constraints_lcg(self):
        """Test get_violated_constraints method for LCG (version 1)."""
        self.violated_constraints1(LCG)

    def test_violated_constraints_vcg(self):
        """Test get_violated_constraints method for VCG (version 1)."""
        self.violated_constraints1(VCG)

    def violated_constraints2(self, rep: SATRepresentation):
        """Test get_violated_constraints method (version 2)."""
        unsatisfiable_str = "p cnf 4 16 \n"
        lits = (np.arange(4) + 1).tolist()
        for bitstring in all_bitstrings(4):
            counter_unsat = ((bitstring - 0.5) * 2).astype(np.int32) * lits
            counter_unsat = (
                np.array2string(counter_unsat).strip("[").strip("]") + " 0 \n "
            )
            unsatisfiable_str += counter_unsat

        cnf = CNF(from_string=unsatisfiable_str)
        problem = get_problem_from_cnf(cnf, rep)

        assert list(problem.params) == [4, 16, 4]

        # assert that always exactly one constraint is violated
        self.assertTrue(
            np.all(
                [
                    jnp.sum(rep.get_violated_constraints(problem, assignment))
                    for assignment in all_bitstrings(4)
                ]
                == np.ones(2**4)
            )
        )

    def test_violated_constraints2_lcg(self):
        """Test get_violated_constraints method for LCG (version 2)."""
        self.violated_constraints2(LCG)

    def test_violated_constraints2_vcg(self):
        """Test get_violated_constraints method for VCG (version 2)."""
        self.violated_constraints2(VCG)

    def inequal_clause_lengths(self, rep: SATRepresentation):
        """Test inequal clause lengths."""
        string = "p cnf 3 2 \n 1 2 0 \n 3 0"
        cnf = CNF(from_string=string)
        problem = get_problem_from_cnf(cnf, rep)
        assert list(problem.params) == [3, 2, 2]
        self.assertEqual(
            rep.get_violated_constraints(problem, np.array([0, 0, 1])).tolist(), [1, 0]
        )
        self.assertEqual(
            rep.get_violated_constraints(problem, np.array([0, 1, 1])).tolist(), [0, 0]
        )
        self.assertEqual(
            rep.get_violated_constraints(problem, np.array([0, 1, 0])).tolist(), [0, 1]
        )

    def test_inequal_clause_lengths_lcg(self):
        """Test inequal clause lengths for LCG."""
        self.inequal_clause_lengths(LCG)

    def test_inequal_clause_lengths_vcg(self):
        """Test inequal clause lengths for VCG."""
        self.inequal_clause_lengths(VCG)


# %%
