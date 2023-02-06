# Copied from https://github.com/tsa87/walk-sat-solver

import random


class SAT:
    """
    A class used to represent a propositional satisfiability (SAT) problem
    Attributes
    ----------
    clauses : [[variables]]
        a list of list of variables that represent or-clauses and-ed in CNF
    variables: set(int)
        a set variables identified by natural numbers
    Methods
    -------
    load_from_file(file_path)
        loads SAT problem from a .txt file in miniSAT format
    solve(p, max_flips)
        solve SAT problem using walk_sat algorithm.
    """

    def __init__(self, clauses=[], variables=set()):
        self.clauses = clauses
        self.variables = variables

    def load_from_file(self, file_path):
        file = open(file_path, "r")
        _ = file.readline()
        configs = file.readline().split()
        num_var, num_clauses = int(configs[-2]), int(configs[-1])

        for i in range(num_clauses):
            tokens = [int(token) for token in file.readline().split()]

            if tokens[-1] != 0:
                raise Exception("clause lines should end with 0")

            self.clauses.append(tokens[:-1])
            for var in tokens[:-1]:
                self.variables.add(abs(var))

    def walk_sat(self, p, max_flips):
        # 1. Generate a random model
        model = {var: random.choice([True, False]) for var in self.variables}
        sat_clauses, unsat_clauses = self.check_assignment(model)

        for i in range(max_flips):
            if not unsat_clauses:
                return model

            rand_clause = random.choice(unsat_clauses)
            if random.uniform(0, 1) < p:
                chosen_var = abs(random.choice(rand_clause))
            else:

                def sat_count(var):
                    model[var] = not var
                    count = len(
                        [
                            clause
                            for clause in self.clauses
                            if self.check_clause(clause, model)
                        ]
                    )
                    model[var] = not var
                    return count

                max_sat = 0
                for var in rand_clause:
                    num_sat = sat_count(abs(var))
                    if num_sat > max_sat:
                        chosen_var = abs(var)
                        max_sat = num_sat

            model[chosen_var] = not model[chosen_var]
            sat_clauses, unsat_clauses = self.check_assignment(model)
        return None

    def check_assignment(self, model):
        sat_clauses = []
        unsat_clauses = []
        for clause in self.clauses:
            (sat_clauses if self.check_clause(clause, model) else unsat_clauses).append(
                clause
            )
        return sat_clauses, unsat_clauses

    def check_clause(self, clause, model):
        for token in clause:
            if token > 0 and model[token] == True:
                return True
            if token < 0 and model[abs(token)] == False:
                return True
        return False
