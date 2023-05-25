import sys

sys.path.append("../../")

from collections import namedtuple
from os.path import join, exists, basename
from os import mkdir

import glob
import gzip
import jraph
import nnf
import numpy as np
import pickle
import jax.numpy as jnp
from func_timeout import func_timeout, FunctionTimedOut
from jax import vmap
from pysat.formula import CNF
from torch.utils import data

from python.src.sat_representations import SATRepresentation
from python.src.sat_instances import get_problem_from_cnf
from python.src.random_walk import number_of_violated_constraints

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
                    candidates = np.array(list(solution_dict.values()), dtype=int).reshape(
                                                1, -1
                                            )  # (1, n_node)
                else:
                    candidates = np.array([solution_dict])


        padded_candidates = self.representation.get_padded_candidate(
            candidates, self.max_n_node
        )
        violated_constraints = vmap(
            self.representation.get_violated_constraints, in_axes=(None, 0), out_axes=0
        )(problem, candidates)
        energies = jnp.sum(violated_constraints, axis=1)  # (n_candidates,)
        return problem, (padded_candidates, energies)


def collate_fn(batch):
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


def create_candidates(data_dir, sample_size, threshold):
    solved_instances = glob.glob(join(data_dir, "*_sol.pkl"))
    for g in solved_instances:
        with open(g, "rb") as f:
            p = pickle.load(f)
        n = np.array(list(p.values()), dtype=bool)
        samples = sample_candidates(n, sample_size - 1, threshold)
        samples = np.concatenate((np.reshape(n, (1, len(n))), samples), axis=0)
        name = g.split("_sol.pkl")[0]
        with open(name + "_samples_sol.npy", "wb") as f:
            np.save(f, samples)


def sample_candidates(original, sample_size, threshold):
    np.random.seed(sum(original))
    condition = np.random.random((sample_size, original.shape[0])) < threshold
    return np.where(condition, np.invert(original), original)
