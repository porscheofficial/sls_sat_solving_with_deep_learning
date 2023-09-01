import sys

sys.path.append("../../")

from collections import namedtuple
from os.path import join, exists, basename
from os import mkdir
from functools import partial
import glob
import gzip
import jraph
import jax
import nnf
import numpy as np
import pickle
import jax.numpy as jnp
from func_timeout import func_timeout, FunctionTimedOut
from jraph._src import utils
from jax import vmap
from pysat.formula import CNF
from torch.utils import data

from python.src.sat_instances import SATProblem, SATRepresentation
from python.src.sat_representations import SATRepresentation, LCG, VCG
from python.src.sat_instances import get_problem_from_cnf

MAX_TIME = 20

SATInstanceMeta = namedtuple("SATInstanceMeta", ("name", "n", "m"))


class SATTrainingDataset(data.Dataset):
    def __init__(
        self,
        data_dir,
        representation,
        already_unzipped=True,
        return_candidates=True,
        include_constraint_graph=False,
    ):
        self.return_candidates = return_candidates
        self.data_dir = data_dir
        self.already_unzipped = already_unzipped
        self.representation: SATRepresentation = representation
        self.include_constraint_graph = include_constraint_graph
        solved_instances = glob.glob(join(data_dir, "*_sol.pkl"))

        self.instances = []
        edges_list = []
        n_nodes_list = []
        for f in solved_instances:
            name = f.split("_sol.pkl")[0]
            problem_file = self._get_problem_file(name)
            cnf = CNF(from_string=problem_file.read())
            n, m = cnf.nv, len(cnf.clauses)
            instance = SATInstanceMeta(name, n, m)
            n_nodes_list.append(self.representation.get_n_nodes(cnf))
            edges_list.append(self.representation.get_n_edges(cnf))
            self.instances.append(instance)
        self.max_n_node = n + 2 if (n := max(n_nodes_list)) % 2 == 0 else n + 1
        self.max_n_edge = max(edges_list)

    def __len__(self):
        return len(self.instances)

    def _get_problem_file(self, name):
        if self.already_unzipped:
            return open(name + ".cnf", "rt")
        else:
            return gzip.open(name + ".cnf.gz", "rt")

    def get_unpadded_problem(self, idx):
        instance_name = self.instances[idx].name
        problem_file = self._get_problem_file(instance_name)
        return get_problem_from_cnf(
            cnf=CNF(from_string=problem_file.read()), representation=self.representation
        )

    def __getitem__(self, idx):
        instance = self.instances[idx]
        instance_name = instance.name
        problem_file = self._get_problem_file(instance_name)
        problem = get_problem_from_cnf(
            cnf=CNF(from_string=problem_file.read()),
            pad_nodes=self.max_n_node,
            pad_edges=self.max_n_edge,
            representation=self.representation,
            include_constraint_graph=self.include_constraint_graph,
        )
        if self.return_candidates:
            # return not just solution but also generated candidates
            target_name = instance_name + "_samples_sol.npy"
            candidates = np.load(target_name)  # (n_candidates, n_node)
            candidates = np.array(candidates, dtype=int)
        else:
            # return only solution
            target_name = instance_name + "_sol.pkl"
            with open(target_name, "rb") as f:
                solution_dict = pickle.load(f)
            if type(solution_dict) == dict:
                print("dict", solution_dict)
                solution_dict = [x for (_, x) in solution_dict.items()]
            if type(solution_dict) == (list or np.array):
                if 2 in solution_dict or -2 in solution_dict:
                    solution_dict = np.array(solution_dict, dtype=float)
                    solution_dict = [int(np.sign(x) + 1) / 2 for x in solution_dict]
            candidates = np.array([solution_dict])
            # with open(target_name, "rb") as f:
            #    solution_dict = pickle.load(f)
            #    if type(solution_dict) == dict:
            #        candidates = np.array(
            #            list(solution_dict.values()), dtype=int
            #        ).reshape(
            #            1, -1
            #        )  # (1, n_node)
            #    else:
            #        candidates = np.array([solution_dict])

        padded_candidates = self.representation.get_padded_candidate(
            candidates, self.max_n_node
        )
        violated_constraints = vmap(
            self.representation.get_violated_constraints, in_axes=(None, 0), out_axes=0
        )(problem, candidates)
        energies = jnp.sum(violated_constraints, axis=1)  # (n_candidates,)
        assert energies[0] == 0
        return problem, (padded_candidates, energies)


def collate_fn(batch):
    """This function does batching. See description below what you input. It returns the batched data items as indicated below.

    Args:
        batch (_type_): batch consisting of (problems,tuples) = zip(*batch). Then (candidates, energies) = zip(*tuples) and masks, graphs, constraint_graphs, constraint_mask = zip(
        *((p.mask, p.graph, p.constraint_graph, p.constraint_mask) for p in problems)
    )

    Raises:
        ValueError: either all items must have a constraint graph or none of them. If this is not the case, a value error is raised.

    Returns:
        tuple: returns the following
                (
                    batched_masks,
                    batched_graphs,
                    batched_constraint_graphs,
                    batched_constraint_masks,
                ), (
                    batched_candidates,
                    batched_energies,
                )
    """
    problems, tuples = zip(*batch)
    candidates, energies = zip(*tuples)
    masks, graphs, constraint_graphs, constraint_mask = zip(
        *((p.mask, p.graph, p.constraint_graph, p.constraint_mask) for p in problems)
    )
    batched_masks = np.concatenate(masks)

    # we expect either all data items to have a constraint graph or none
    if all(g is None for g in constraint_graphs):
        batched_constraint_graphs = None
        batched_constraint_masks = None
    elif not any(g is None for g in constraint_graphs):
        batched_constraint_graphs = jraph.batch(constraint_graphs)
        batched_constraint_masks = np.concatenate(constraint_mask)
    else:
        raise ValueError("Either all data items must have a constraint graph or none")

    batched_graphs = jraph.batch(graphs)
    batched_candidates = np.vstack([c.T for c in candidates])
    batched_energies = np.vstack(
        [np.repeat([e], np.shape(c)[1], axis=0) for (e, c) in zip(energies, candidates)]
    )

    return (
        batched_masks,
        batched_graphs,
        batched_constraint_graphs,
        batched_constraint_masks,
    ), (
        batched_candidates,
        batched_energies,
    )


class JraphDataLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def number_of_violated_constraints(
    problem: SATProblem, assignment, representation: SATRepresentation
):
    """helper function to get the number of violated constraints for a SATProblem and an assignment for a given SATRepresentation

    Args:
        problem (SATProblem): SAT problem we want to look at
        assignment (array): current assignment of variables we want to check for
        representation (SATRepresentation): SATRepresentation -> either LCG or VCG

    Returns:
        int: number of violated constraints by current assignment for the problem at hand
    """
    match representation:
        case VCG:
            return number_of_violated_constraints_VCG(problem, assignment)
        case LCG:
            return number_of_violated_constraints_LCG(problem, assignment)


def number_of_violated_constraints_VCG(problem: SATProblem, assignment):
    # currently not implemented
    pass
    return np.sum(
        SATRepresentation.VCG.violated_constraints(problem, assignment).astype(int),
        axis=0,
    )


@partial(jax.jit, static_argnames=("problem",))
def number_of_violated_constraints_LCG(problem: SATProblem, assignment):
    def one_hot(x, k, dtype=jnp.float32):
        """Create a one-hot encoding of x of size k."""
        return jnp.array(x[:, None] == jnp.arange(k), dtype)

    graph = problem.graph
    n, m, k = problem.params
    senders = graph.senders[:-n]
    receivers = graph.receivers[:-n]
    new_assignment = jnp.ravel(one_hot(assignment, 2))
    edge_is_satisfied = jnp.ravel(
        new_assignment[None].T[senders].T
    )  # + np.ones(len(senders)), 2)
    number_of_literals_satisfied = utils.segment_sum(
        data=edge_is_satisfied, segment_ids=receivers, num_segments=2 * n + m
    )[2 * n :]
    clause_is_unsat = jnp.where(number_of_literals_satisfied > 0, 0, 1)
    return jnp.sum(clause_is_unsat)


def timed_solve(max_time, p):
    try:
        return func_timeout(max_time, nnf.kissat.solve, args=(p,))
    except FunctionTimedOut:
        print(f"Could not be solved within time limit of {max_time} seconds")
    return None


def create_solutions_from_cnf(path, time_limit=MAX_TIME):
    return create_solutions(path, time_limit, suffix="*.cnf", open_util=open)


def create_solutions_from_gzip(path, time_limit=MAX_TIME):
    return create_solutions(path, time_limit, suffix="*.cnf.gz", open_util=gzip.open)


def create_solutions(path, time_limit, suffix, open_util):
    regex = join(path, suffix)
    for f in glob.glob(regex):
        print(f"processing {f}")
        root = f.split(".cnf")[0]
        solved_target_name = root + "_sol.pkl"
        unsolved_target_name = root + "_unsol.pkl"
        solved_target_name = join(solved_target_name)
        unsolved_target_name = join(unsolved_target_name)
        if exists(solved_target_name) or exists(unsolved_target_name):
            print("solution file already exists")
            continue
        with open_util(f, mode="rt") as fl:
            p = nnf.dimacs.load(fl)
            s = timed_solve(time_limit, p)
            if not s:
                with open(unsolved_target_name, "wb") as out:
                    pickle.dump(s, out)
            else:
                with open(solved_target_name, "wb") as out:
                    pickle.dump(s, out)
                    print(f"written solution for {root}")


def create_candidates(data_dir, sample_size: int, threshold):
    """helper function to create candidates from solution -> used for Gibbs Loss

    Args:
        data_dir (str): path to data directory where you want to create candidates
        sample_size (int): number of candidates that are created
        threshold (float): float ranging from 0 to 1. This is the probability that a spin flip is executed on a variable in the solution string
    """
    solved_instances = glob.glob(join(data_dir, "*_sol.pkl"))
    for g in solved_instances:
        with open(g, "rb") as f:
            p = pickle.load(f)
        if type(p) == dict:
            print("dict", p)
            p = [x for (_, x) in p.items()]
        if type(p) == list or type(p) == np.array:
            if 2 in p or -2 in p:
                p = np.array(p, dtype=float)
                p = [int(np.sign(x) + 1) / 2 for x in p]
        print(p)
        n = np.array(p, dtype=bool)
        print(n)
        samples = sample_candidates(n, sample_size - 1, threshold)
        samples = np.concatenate((np.reshape(n, (1, len(n))), samples), axis=0)
        name = g.split("_sol.pkl")[0]
        with open(name + "_samples_sol.npy", "wb") as f:
            np.save(f, samples)


def sample_candidates(original, sample_size, threshold):
    """helper function to execute the sampling of one candidate

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
