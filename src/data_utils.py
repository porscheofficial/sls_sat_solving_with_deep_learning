from collections import namedtuple
from os.path import join, exists

import glob
import gzip
import jraph
import nnf
import numpy as np
import pickle
from func_timeout import func_timeout, FunctionTimedOut
from jax import vmap
from jax.numpy import asarray
from pysat.formula import CNF
from torch.utils import data

from constraint_problems import get_problem_from_cnf
from random_walk import number_of_violated_constraints

MAX_TIME = 20

SATInstanceMeta = namedtuple("SATInstanceMeta", ("name", "n", "m", "n_edges"))


class SATTrainingDataset(data.Dataset):
    def __init__(self, data_dir, already_unzipped=True, return_candidates=True):
        self.return_candidates = return_candidates
        self.data_dir = data_dir
        self.already_unzipped = already_unzipped
        solved_instances = glob.glob(join(data_dir, "*_sol.pkl"))

        self.instances = []
        for f in solved_instances:
            name = f.split("_sol.pkl")[0]
            problem_file = self._get_problem_file(name)
            cnf = CNF(from_string=problem_file.read())
            instance = SATInstanceMeta(
                name, cnf.nv, len(cnf.clauses), sum(len(c) for c in cnf.clauses)
            )
            self.instances.append(instance)
        self.max_n_node = max(i.n + i.m for i in self.instances)
        self.max_n_edge = max(i.n_edges for i in self.instances)

    def __len__(self):
        return len(self.instances)

    def solution_dict_to_array(self, solution_dict):
        return np.pad(
            np.array(list(solution_dict.values()), dtype=int),
            (0, self.max_n_node - len(solution_dict)),
        )

    def _get_problem_file(self, name):
        if self.already_unzipped:
            return open(name + ".cnf", "rt")
        else:
            return gzip.open(name + ".cnf.gz", "rt")

    def __getitem__(self, idx):
        instance_name = self.instances[idx].name
        problem_file = self._get_problem_file(instance_name)
        problem = get_problem_from_cnf(
            cnf=CNF(from_string=problem_file.read()),
            pad_nodes=self.max_n_node,
            pad_edges=self.max_n_edge,
        )
        N = len(problem.mask)  # total number of nodes in (padded) graph
        n, _, _ = problem.params  # number of variables nodes in instance

        if self.return_candidates:
            # return not just solution but also generated candidates
            target_name = instance_name + "_samples_sol.npy"
            candidates = np.load(
                target_name
            )  # np.array which stores candidates and solution to problem

            padded_candidates = np.pad(
                candidates,
                pad_width=((0, 0), (0, N - n)),
            )
            energies = vmap(
                number_of_violated_constraints, in_axes=(None, 0), out_axes=0
            )(problem, candidates)
            return problem, (padded_candidates, energies)
        else:
            # return only solution
            target_name = instance_name + "_sol.pkl"
            with open(target_name, "rb") as f:
                solution_dict = pickle.load(f)
                candidates = self.solution_dict_to_array(solution_dict)
            energies = number_of_violated_constraints(problem, candidates)
            # candidates already padded inside solution_dict_to_array but repeated here for transparency
            padded_candidates = np.pad(
                candidates,
                pad_width=(0, N - n),
            )
            return problem, (padded_candidates, energies)


def collate_fn(batch):
    problems, tuples = zip(*batch)
    candidates, energies = zip(*tuples)
    masks, graphs = zip(*((p.mask, p.graph) for p in problems))
    batched_masks = np.concatenate(masks)
    batched_graphs = jraph.batch(graphs)
    batched_candidates = np.vstack([c.T for c in candidates])
    batched_energies = np.vstack(
        [np.repeat([e], len(m), axis=0) for (e, m) in zip(energies, masks)]
    )
    return (batched_masks, batched_graphs), (batched_candidates, batched_energies)


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
        solved_target_name = join("processed", "solved", solved_target_name)
        unsolved_target_name = join("processed", "unsolved", unsolved_target_name)
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
