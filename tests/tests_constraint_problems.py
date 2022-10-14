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
        contradiction= "p cnf 1 2 \n 1 0 \n -1 0"
        return CNF(from_string=contradiction)

    def test_problem_parsing(self):
        problem = get_problem_from_cnf(self.instance_simple_tautology())
        self.assertEquals(list(problem.meta.values()), [1, 1, 2])

    def test_violated_constraints(self):
        assignments = all_bitstrings(1)
        problem = get_problem_from_cnf(self.instance_simple_tautology())
        # violated_constraints returns False when a constraint is *not* violated, i.e. it is satisfied.
        self.assertFalse(any([violated_constraints(problem, a) for a in assignments]))

        problem = get_problem_from_cnf(self.instance_simple_contradiction())
        self.assertEqual([jnp.sum(violated_constraints(problem, a)) for a in assignments], [1, 1])

        unsatisfiable_str = "p cnf 4 16 \n"
        lits = (np.arange(4) + 1).tolist()
        for c in all_bitstrings(4):
            s = ((c - 0.5) * 2).astype(np.int32) * lits
            s = np.array2string(s).strip("[").strip("]") + " 0 \n "
            unsatisfiable_str += s

        cnf = CNF(from_string=unsatisfiable_str)
        problem = get_problem_from_cnf(cnf)

        assert list(problem.meta.values()) == [4, 16, 4]

        # assert that always exactly one constraint is violated
        self.assertTrue(all([jnp.sum(violated_constraints(problem, a)) for a in all_bitstrings(4)] == np.ones(2**4)))

#%%
