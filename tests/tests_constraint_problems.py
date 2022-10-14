import unittest
from pysat.formula import CNF
import jax.numpy as jnp
import numpy as np

from src.constraint_problems import violated_constraints, all_bitstrings, get_problem_from_cnf


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
            get_problem_from_cnf(self.instance_simple_tautology())
        except AssertionError:
            pass
        self.assertRaises(AssertionError)

    def test_problem_parsing(self):
        problem = get_problem_from_cnf(self.instance_simple_contradiction())
        self.assertEquals(problem.params, [1, 2, 1])

        string = "p cnf 3 2 \n 1 2 0 \n 3 0"
        cnf = CNF(from_string=string)
        problem = get_problem_from_cnf(cnf)
        assert problem.params == [3, 2, 2]
        g = problem.graph
        assert len(g.nodes) == g.n_node
        assert len(g.receivers) == len(g.senders)
        assert len(g.senders) == len(g.edges)
        assert len(g.edges) == g.n_edge

    def test_violated_constraints1(self):
        assignments = all_bitstrings(1)
        problem = get_problem_from_cnf(self.instance_simple_contradiction())
        self.assertEqual([jnp.sum(violated_constraints(problem, a)) for a in assignments], [1, 1])

    def test_violated_constraints2(self):
        unsatisfiable_str = "p cnf 4 16 \n"
        lits = (np.arange(4) + 1).tolist()
        for c in all_bitstrings(4):
            s = ((c - 0.5) * 2).astype(np.int32) * lits
            s = np.array2string(s).strip("[").strip("]") + " 0 \n "
            unsatisfiable_str += s

        cnf = CNF(from_string=unsatisfiable_str)
        problem = get_problem_from_cnf(cnf)

        assert list(problem.params) == [4, 16, 4]

        # assert that always exactly one constraint is violated
        self.assertTrue(all([jnp.sum(violated_constraints(problem, a)) for a in all_bitstrings(4)] == np.ones(2 ** 4)))

    def test_inequal_clause_lengths(self):
        string = "p cnf 3 2 \n 1 2 0 \n 3 0"
        cnf = CNF(from_string=string)
        problem = get_problem_from_cnf(cnf)
        assert problem.params == [3, 2, 2]
        self.assertEqual(violated_constraints(problem, np.array([0, 0, 1])).tolist(), [1, 0])
        self.assertEqual(violated_constraints(problem, np.array([0, 1, 1])).tolist(), [0, 0])
        self.assertEqual(violated_constraints(problem, np.array([0, 1, 0])).tolist(), [0, 1])
# %%
