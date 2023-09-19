"""Contains functions to generate random SAT instances."""
import pickle
import random
import cnfgen
from pysat.formula import CNF
from pysat.solvers import Glucose3


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


def generate_random_kcnf(k_locality, n_variables, n_clauses, path, timeout=100):
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
    while current_time <= timeout and not sol:
        current_time += 1
        # print(t)
        cnf = cnfgen.RandomKCNF(k_locality, n_variables, n_clauses)
        cnf = cnf.to_dimacs()
        cnf = CNF(from_string=cnf)
        solver_result = Glucose3(cnf)
        if solver_result.solve():
            sol = solver_result.get_model()
            cnf.to_file(path + ".cnf")
            with open(path + "_sol.pkl", "wb") as file:
                pickle.dump(sol, file)
    if not sol:
        print(
            "no satisfiable random_KCNF problem found for (n,k,m)=("
            + str(n_variables)
            + ","
            + str(k_locality)
            + ","
            + str(n_clauses)
            + ")"
        )


def generate_dataset_random_kcnf(
    k_locality, n_variables_list, alpha, num_samples, path, vary_percent=0, timeout=100
):
    """Generate a random_KCNF dataset.

    Args:
        k_locality (int): locality of clauses
        n_variables_list (list): list of number of variables that should be used for generating
        alpha (float): float describing the desired density of KCNF
        num_samples (int): number of samples generated per value in n_list
        path (str): path of where dataset should be saved
        vary_percent (float, optional): describes by how much we vary m at maximum
        timeout (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    for n_variables in n_variables_list:
        for _ in range(num_samples):
            vary = 2 * (1 / 2 - random.random()) * vary_percent
            n_clauses = int((1 + vary) * alpha * n_variables)
            index = str(random.randint(0, 10000000))
            params = (
                str(k_locality) + "_" + str(n_variables) + "_" + str(n_clauses) + "_"
            )
            generate_random_kcnf(
                k_locality,
                n_variables,
                n_clauses,
                path=path + "random_KCNF" + params + index,
                timeout=timeout,
            )


def generate_ramsey(s_parameter, k_parameter, n_parameter, path, timeout=100):
    """Generate a single Ramsey formula.

    Ramsey number r(s,k) > N
    This formula, given s, k, and N, claims that there is some graph with N vertices which has neither
    independent sets of size s nor cliques of size k.
    It turns out that there is a number r(s,k) so that every graph with at least r(s,k) vertices must
    contain either one or the other. Hence the generated formula is satisfiable if and only if r(s,k)>N

    Args:
        s_parameter (int): independent set size
        k_parameter (int): clique size
        n_parameter (int): number of vertices
        path (str): path and name how instance should be saved
        timeout (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    time = 0
    sol = False
    while time <= timeout and not sol:
        time += 1
        # print(t)
        cnf = cnfgen.RamseyNumber(s_parameter, k_parameter, n_parameter)
        cnf = cnf.to_dimacs()
        cnf = CNF(from_string=cnf)
        solver_result = Glucose3(cnf)
        if solver_result.solve():
            sol = solver_result.get_model()
            cnf.to_file(path + ".cnf")
            with open(path + "_sol.pkl", "wb") as file:
                pickle.dump(sol, file)
    if not sol:
        print(
            "no satisfiable Ramsey problem found for (s,k,N)=("
            + str(s_parameter)
            + ","
            + str(k_parameter)
            + ","
            + str(n_parameter)
            + ")"
        )


def generate_dataset_ramsey(s_list, k_list, n_list, num_samples, path, timeout=100):
    """Generate a Ramsey dataset.

    Ramsey number r(s,k) > N
    This formula, given s, k, and N, claims that there is some graph with N vertices which has neither
    independent sets of size s nor cliques of size k.
    It turns out that there is a number r(s,k) so that every graph with at least r(s,k) vertices must
    contain either one or the other. Hence the generated formula is satisfiable if and only if r(s,k)>N

    Args:
        s_list (int): independent set size
        k_list (int): clique size
        n_list (int): number of vertices
        num_samples (int): number of samples generated per set of parameters
        path (str): path and name how instance should be saved
        timeout (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    for s_parameter in s_list:
        for k_parameter in k_list:
            for n_parameter in n_list:
                for _ in range(num_samples):
                    index = str(random.randint(0, 10000000))
                    params = (
                        str(s_parameter)
                        + "_"
                        + str(k_parameter)
                        + "_"
                        + str(n_parameter)
                        + "_"
                    )
                    generate_ramsey(
                        s_parameter,
                        k_parameter,
                        n_parameter,
                        path + "ramsey" + params + index,
                        timeout=timeout,
                    )


def generate_van_der_waerden(
    interval_size, k1_parameter, k2_parameter, path, timeout=100
):
    """Generate a single VanDerWaerden formula.

    NOTE: tbf with details

    Args:
        interval_size (int): size of interval
        k1_parameter (int): length of arithmetic progressions of color 1
        k2_parameter (int): length of arithmetic progressions of color 2
        path (str): path and name how instance should be saved
        timeout (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    time = 0
    sol = False
    while time <= timeout and not sol:
        time += 1
        # print(t)
        cnf = cnfgen.VanDerWaerden(interval_size, k1_parameter, k2_parameter)
        cnf = cnf.to_dimacs()
        cnf = CNF(from_string=cnf)
        solver_result = Glucose3(cnf)
        if solver_result.solve():
            sol = solver_result.get_model()
            cnf.to_file(path + ".cnf")
            with open(path + "_sol.pkl", "wb") as file:
                pickle.dump(sol, file)
    if not sol:
        print(
            "no satisfiable VanDerWaerden problem found for (N, k1, k2)=("
            + str(interval_size)
            + ","
            + str(k1_parameter)
            + ","
            + str(k2_parameter)
            + ")"
        )


def generate_dataset_van_der_waerden(
    interval_size_list, k1_list, k2_list, num_samples, path, timeout=100
):
    """Generate a dataset containing VanDerWaerden formulas.

    NOTE: tbf with details

    Args:
        interval_size_list (list): list of size of interval
        k1_list (list): list of length of arithmetic progressions of color 1
        k2_list (list): list of length of arithmetic progressions of color 2
        num_samples (int): number of samples generated per set of parameters
        path (str): path and name how instance should be saved
        TIMEOUT (int, optional): how often we try to generate a satisfying formula until we return no instance. Defaults to 100.
    """
    for interval_size in interval_size_list:
        for k1_parameter in k1_list:
            for k2_parameter in k2_list:
                for _ in range(num_samples):
                    index = str(random.randint(0, 10000000))
                    params = (
                        str(interval_size)
                        + "_"
                        + str(k1_parameter)
                        + "_"
                        + str(k2_parameter)
                        + "_"
                    )
                    generate_van_der_waerden(
                        interval_size,
                        k1_parameter,
                        k2_parameter,
                        path + "VanDerWaerden" + params + index,
                        timeout=timeout,
                    )
