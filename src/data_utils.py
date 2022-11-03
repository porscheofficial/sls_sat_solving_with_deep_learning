import glob
import os

import nnf
import gzip
import pickle
from func_timeout import func_timeout, FunctionTimedOut
from os.path import exists

MAX_TIME = 20


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
    regex = os.path.join(path, suffix)
    for f in glob.glob(regex):
        print(f"processing {f}")
        root = os.path.splitext(f)[0]
        solved_target_name = root + "_sol.pkl"
        unsolved_target_name = root + "_unsol.pkl"
        # solved_target_name = os.path.join(path, solved_target_name)
        # unsolved_target_name = os.path.join(path, unsolved_target_name)
        if exists(solved_target_name) or exists(unsolved_target_name):
            print("solution file already exists")
            continue
        with open_util(f, mode='rt') as fl:
            p = nnf.dimacs.load(fl)
            s = timed_solve(time_limit, p)
            if not s:
                with open(unsolved_target_name, "wb") as out:
                    pickle.dump(s, out)
            else:
                with open(solved_target_name, "wb") as out:
                    pickle.dump(s, out)
                    print(f"written solution for {root}")
