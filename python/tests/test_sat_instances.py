import unittest
from pysat.formula import CNF
import jax.numpy as jnp
import numpy as np
import sys

sys.path.append("../../")

from python.src.sat_representations import VCG, LCG, SATRepresentation
from python.src.sat_instances import (
    all_bitstrings,
    get_problem_from_cnf,
)


class TestConstraintProblemUtils(unittest.TestCase):
    @staticmethod
    def instance_simple_tautology():
        tautology = "p cnf 1 1 \n 1 -1 0"
        return CNF(from_string=tautology)

    @staticmethod
    def instance_simple_contradiction():
        contradiction = "p cnf 1 2 \n 1 0 \n -1 0"
        return CNF(from_string=contradiction)

    def test_forbid_multiple_variable_occurrences_in_single_clause(self):
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
        problem = get_problem_from_cnf(self.instance_simple_contradiction(), rep)
        self.assertEquals(problem.params, (1, 2, 1))

        string = "p cnf 3 2 \n 1 2 0 \n 3 0"
        cnf = CNF(from_string=string)
        problem = get_problem_from_cnf(cnf, rep)
        assert problem.params == (3, 2, 2)
        g = problem.graph
        assert len(g.nodes) == g.n_node
        assert len(g.receivers) == len(g.senders)
        assert len(g.senders) == len(g.edges)
        assert len(g.edges) == g.n_edge

    def test_problem_parsing_lcg(self):
        self.problem_parsing(LCG)

    def test_problem_parsing_vcg(self):
        self.problem_parsing(VCG)

    def violated_constraints1(self, rep: SATRepresentation):
        assignments = all_bitstrings(1)
        problem = get_problem_from_cnf(self.instance_simple_contradiction(), rep)
        self.assertEqual(
            [np.sum(rep.get_violated_constraints(problem, a)) for a in assignments],
            [1, 1],
        )

    def test_violated_constraints_lcg(self):
        self.violated_constraints1(LCG)

    def test_violated_constraints_vcg(self):
        self.violated_constraints1(VCG)

    def violated_constraints2(self, rep: SATRepresentation):
        unsatisfiable_str = "p cnf 4 16 \n"
        lits = (np.arange(4) + 1).tolist()
        for c in all_bitstrings(4):
            s = ((c - 0.5) * 2).astype(np.int32) * lits
            s = np.array2string(s).strip("[").strip("]") + " 0 \n "
            unsatisfiable_str += s

        cnf = CNF(from_string=unsatisfiable_str)
        problem = get_problem_from_cnf(cnf, rep)

        assert list(problem.params) == [4, 16, 4]

        # assert that always exactly one constraint is violated
        self.assertTrue(
            [
                jnp.sum(rep.get_violated_constraints(problem, a))
                for a in all_bitstrings(4)
            ]
            == np.ones(2**4)
        )

    def test_violated_constraints2_lcg(self):
        self.violated_constraints2(LCG)

    def test_violated_constraints2_vcg(self):
        self.violated_constraints2(VCG)

    def inequal_clause_lengths(self, rep: SATRepresentation):
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
        self.inequal_clause_lengths(LCG)

    def test_inequal_clause_lengths_vcg(self):
        self.inequal_clause_lengths(VCG)


# %%
