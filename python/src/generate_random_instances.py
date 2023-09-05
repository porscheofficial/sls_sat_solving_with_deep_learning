"""Contains functions to generate random SAT instances."""
import numpy as np
import cnfgen
from pysat.formula import CNF
from pysat.solvers import Glucose3
import pickle
import random
import glob
from os.path import join


def get_cnf_from_file(path):
    """Get a cnf object from a file.

    Args:
        path (str): path of cnf object you want to load

    Returns:
        CNF: cnf object that was loaded
    """
    if path[-2:] == "gz":
        cnf = CNF()
        cnf.from_file(path, compressed_with="gzip")
    else:
        cnf = CNF(from_file=path)
    return cnf


def create_candidates_with_sol(data_dir, sample_size, threshold):
    """Create candidates from solution -> used for Gibbs Loss.

    Args:
        data_dir (str): path to data directory where you want to create candidates
        sample_size (int): number of candidates that are created
        threshold (float): float ranging from 0 to 1. This is the probability that a spin flip is executed on a variable in the solution string
    """
    solved_instances = glob.glob(join(data_dir, "*_sol.pkl"))
    for solved_instance in solved_instances:
        with open(solved_instance, "rb") as file:
            solution = pickle.load(file)
        if type(solution) == dict:
            print("dict", solution)
            solution = [assignment_x for (_, assignment_x) in solution.items()]
        if type(solution) == list or type(solution) == np.array:
            if 2 in solution or -2 in solution:
                solution = np.array(solution, dtype=float)
                solution = [
                    int(np.sign(assignment_x) + 1) / 2 for assignment_x in solution
                ]
        print(solution)
        solution_boolean = np.array(solution, dtype=bool)
        print(solution_boolean)
        samples = sample_candidates(solution_boolean, sample_size - 1, threshold)
        samples = np.concatenate(
            (np.reshape(solution_boolean, (1, len(solution_boolean))), samples), axis=0
        )
        name = solved_instance.split("_sol.pkl")[0]
        with open(name + "_samples_sol.npy", "wb") as file:
            np.save(file, samples)


def sample_candidates(original, sample_size, threshold):
    """Execute the sampling of one candidate.

    Args:
        original: original solution string that is modified in this function
        sample_size: number of candidates that are created
        threshold: float ranging from 0 to 1. This is the probability that a spin flip is executed on a variable in the solution string

    Returns:
        np.array: returns a matrix containing a set of candidates and the solution itself
    """
    np.random.seed(sum(original))
    condition = np.random.random((sample_size, original.shape[0])) < threshold
    return np.where(condition, np.invert(original), original)


def generate_random_KCNF(k_locality, n_variables, n_clauses, path, timeout=100):
    """Generate a single random_KCNF formula.

    Args:
        k_locality (int): locality of instance
        n_variables (int): number of variables of instance
        n_clauses (int): number of clauses contained in instance
        path (str): path and name how instance should be saved
        timeout (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    current_time = 0
    sol = False
    while current_time <= timeout and sol == False:
        current_time += 1
        # print(t)
        cnf = cnfgen.RandomKCNF(k_locality, n_variables, n_clauses)
        cnf = cnf.to_dimacs()
        cnf = CNF(from_string=cnf)
        solver_result = Glucose3(cnf)
        if solver_result.solve() == True:
            sol = solver_result.get_model()
            cnf.to_file(path + ".cnf")
            with open(path + "_sol.pkl", "wb") as file:
                pickle.dump(sol, file)
    if sol == False:
        print(
            "no satisfiable random_KCNF problem found for (n,k,m)=("
            + str(n_variables)
            + ","
            + str(k_locality)
            + ","
            + str(n_clauses)
            + ")"
        )


def generate_dataset_random_KCNF(
    k_locality, n_variables_list, alpha, num_samples, path, vary_percent=0, TIMEOUT=100
):
    """Generate a random_KCNF dataset.

    Args:
        k_locality (int): locality of clauses
        n_variables_list (list): list of number of variables that should be used for generating
        alpha (float): float describing the desired density of KCNF
        num_samples (int): number of samples generated per value in n_list
        path (str): path of where dataset should be saved
        vary_percent (float, optional): describes by how much we vary m at maximum
        TIMEOUT (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    for n_variables in n_variables_list:
        for _ in range(num_samples):
            vary = 2 * (1 / 2 - random.random()) * vary_percent
            n_clauses = int((1 + vary) * alpha * n_variables)
            index = str(random.randint(0, 10000000))
            params = (
                str(k_locality) + "_" + str(n_variables) + "_" + str(n_clauses) + "_"
            )
            generate_random_KCNF(
                k_locality,
                n_variables,
                m,
                path=path + "random_KCNF" + params + index,
                timeout=TIMEOUT,
            )


def generate_Ramsey(s, k, N, path, TIMEOUT=100):
    """Generate a single Ramsey formula.

    Ramsey number r(s,k) > N
    This formula, given s, k, and N, claims that there is some graph with N vertices which has neither
    independent sets of size s nor cliques of size k.
    It turns out that there is a number r(s,k) so that every graph with at least r(s,k) vertices must
    contain either one or the other. Hence the generated formula is satisfiable if and only if r(s,k)>N

    Args:
        s (int): independent set size
        k (int): clique size
        N (int): number of vertices
        path (str): path and name how instance should be saved
        TIMEOUT (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    t = 0
    sol = False
    while t <= TIMEOUT and sol == False:
        t += 1
        # print(t)
        cnf = cnfgen.RamseyNumber(s, k, N)
        cnf = cnf.to_dimacs()
        cnf = CNF(from_string=cnf)
        g = Glucose3(cnf)
        if g.solve() == True:
            sol = g.get_model()
            cnf.to_file(path + ".cnf")
            with open(path + "_sol.pkl", "wb") as f:
                pickle.dump(sol, f)
    if sol == False:
        print(
            "no satisfiable Ramsey problem found for (s,k,N)=("
            + str(s)
            + ","
            + str(k)
            + ","
            + str(N)
            + ")"
        )


def generate_dataset_Ramsey(s_list, k_list, N_list, num_samples, path, TIMEOUT=100):
    """Generate a Ramsey dataset.

    Ramsey number r(s,k) > N
    This formula, given s, k, and N, claims that there is some graph with N vertices which has neither
    independent sets of size s nor cliques of size k.
    It turns out that there is a number r(s,k) so that every graph with at least r(s,k) vertices must
    contain either one or the other. Hence the generated formula is satisfiable if and only if r(s,k)>N

    Args:
        s (int): independent set size
        k (int): clique size
        N (int): number of vertices
        num_samples (int): number of samples generated per set of parameters
        path (str): path and name how instance should be saved
        TIMEOUT (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    for s in s_list:
        for k in k_list:
            for N in N_list:
                for _ in range(num_samples):
                    index = str(random.randint(0, 10000000))
                    params = str(s) + "_" + str(k) + "_" + str(N) + "_"
                    generate_Ramsey(
                        s, k, N, path + "ramsey" + params + index, TIMEOUT=100
                    )


def generate_VanDerWaerden(N, k1, k2, path, TIMEOUT=100):
    """Generate a single VanDerWaerden formula.

    NOTE: tbf with details

    Args:
        N (int): size of interval
        k1 (int): length of arithmetic progressions of color 1
        k2 (int): length of arithmetic progressions of color 2
        path (str): path and name how instance should be saved
        TIMEOUT (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    t = 0
    sol = False
    while t <= TIMEOUT and sol == False:
        t += 1
        # print(t)
        cnf = cnfgen.VanDerWaerden(N, k1, k2)
        cnf = cnf.to_dimacs()
        cnf = CNF(from_string=cnf)
        g = Glucose3(cnf)
        if g.solve() == True:
            sol = g.get_model()
            cnf.to_file(path + ".cnf")
            with open(path + "_sol.pkl", "wb") as f:
                pickle.dump(sol, f)
    if sol == False:
        print(
            "no satisfiable VanDerWaerden problem found for (N, k1, k2)=("
            + str(N)
            + ","
            + str(k1)
            + ","
            + str(k2)
            + ")"
        )


def generate_dataset_VanDerWaerden(
    N_list, k1_list, k2_list, num_samples, path, TIMEOUT=100
):
    """Generate a dataset containing VanDerWaerden formulas.

    NOTE: tbf with details

    Args:
        N_list (list): list of size of interval
        k1 (list): list of length of arithmetic progressions of color 1
        k2 (list): list of length of arithmetic progressions of color 2
        num_samples (int): number of samples generated per set of parameters
        path (str): path and name how instance should be saved
        TIMEOUT (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    for N in N_list:
        for k1 in k1_list:
            for k2 in k2_list:
                for _ in range(num_samples):
                    index = str(random.randint(0, 10000000))
                    params = str(N) + "_" + str(k1) + "_" + str(k2) + "_"
                    generate_VanDerWaerden(
                        N,
                        k1,
                        k2,
                        path + "VanDerWaerden" + params + index,
                        TIMEOUT=TIMEOUT,
                    )
