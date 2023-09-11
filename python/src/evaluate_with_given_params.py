"""This code lets you use trained models as oracles for SAT solving in the MT and the WalkSAT algorithm."""
import sys
from functools import partial
import numpy as np
import haiku as hk


sys.path.append("../../")
import moser_rust
from data_utils import SATTrainingDataset
from sat_representations import LCG
from model import (
    get_network_definition,
)

SEED = 0


def get_padded_trajs(traj, n_steps):
    """Do padding for trajectories with zeros if it has found a solution before N_STEPS was reached."""
    array_traj = []
    for _, traj_i in enumerate(traj):
        array_traj.append(np.pad(traj_i, (0, n_steps - len(traj_i))))
    return np.array(array_traj)


def load_model_and_test(
    data_path,
    model_path,
    n_steps: int,
    n_runs: int,
    algo_type,
    path_save=False,
    keep_traj=True,
):
    """Run oracle versions of MT and Walksat algorithm on a dataset to evaluate the performance.

    Args:
        data_path (str): path pointing to dataset used for evaluation
        model_path (str): path of the model or "uniform" for running the uniform version
        n_steps (int): number of steps taken by the algorithm
        N_RUNS (int): number of runs executed by the algorithm
        algo_type (str): either "moser" for MT algorithm or "probsat" for the oracle version of WalkSAT
        path_save (bool, optional): path where details of the experiment are saved. Defaults to False.
        keep_traj (bool, optional): decide whether you want to keep the trajectories. Defaults to True.

    Returns:
        list:       Returns the following elements. They are also saved at path_save.
                        [model_path_initialize, model_path_resample],
                        [model_details_list_i, model_details_list_r],
                        n_array,
                        alpha_array,
                        energies_array_mean,
                        energies_array_median,
                        total_steps,
    """
    energies_array_mean = []
    energies_array_median = []

    n_array = []
    alpha_array = []
    total_steps = list([])
    if model_path != "uniform":
        params, model_details = np.load(model_path, allow_pickle=True)
        (
            inv_temp,
            alpha,
            beta,
            gamma,
            mlp_layers,
            graph_representation,
            network_type,
            return_candidates,
        ) = model_details
        print(graph_representation.__name__)
        model_details_list = [
            inv_temp,
            alpha,
            beta,
            gamma,
            mlp_layers,
            graph_representation.__name__,
            network_type,
            return_candidates,
        ]
        include_constraint_graph = beta > 0
        sat_data = SATTrainingDataset(
            data_path,
            graph_representation,
            return_candidates=return_candidates,
            include_constraint_graph=include_constraint_graph,
        )

        network_definition = get_network_definition(
            network_type=network_type, graph_representation=graph_representation
        )
        network_definition = partial(network_definition, mlp_layers=mlp_layers)
        network = hk.without_apply_rng(hk.transform(network_definition))
    else:
        sat_data = SATTrainingDataset(
            data_path, LCG, return_candidates=False, include_constraint_graph=False
        )
        model_details_list = []

    for idx in range(len(sat_data)):
        print("problem ", idx + 1, "of ", len(sat_data))
        problem_path = sat_data.instances[idx].name + ".cnf"
        problem = sat_data.get_unpadded_problem(idx)

        n_array.append(problem.params[0])
        alpha_array.append(problem.params[1] / problem.params[0])
        if model_path != "uniform":
            decoded_nodes = network.apply(params, problem.graph)  # type: ignore[attr-defined]
            n_variables = problem.params[0]
            model_probabilities = graph_representation.get_model_probabilities(
                decoded_nodes, n_variables
            )
            model_probabilities = model_probabilities.ravel()
        else:
            model_probabilities = np.ones(problem.params[0]) / 2

        single_traj_mean = []
        single_traj_median = []
        print(model_path)
        _, _, _, numstep, traj = moser_rust.run_sls_python(
            algo_type,
            problem_path,
            model_probabilities,
            model_probabilities,
            n_steps - 1,
            n_runs,
            SEED,
            keep_traj,
        )

        total_steps.append(numstep)
        if len(traj) != 0:
            traj = get_padded_trajs(traj, n_steps)
            single_traj_mean.append(np.mean(traj, axis=0) / problem.params[1])
            single_traj_median.append(np.median(traj, axis=0) / problem.params[1])
            energies_array_mean.append(
                np.pad(
                    np.array(single_traj_mean)[0],
                    (0, n_steps - len(single_traj_mean[0])),
                )
            )
            energies_array_median.append(
                np.pad(
                    np.array(single_traj_median)[0],
                    (0, n_steps - len(single_traj_median[0])),
                )
            )
    total_steps_array = np.asarray(total_steps)

    if energies_array_mean:
        energies_array_mean = np.mean(energies_array_mean, axis=0)
        energies_array_median = np.mean(energies_array_median, axis=0)
    total_array = [
        [model_path],
        [model_details_list],
        n_array,
        alpha_array,
        energies_array_mean,
        energies_array_median,
        total_steps_array,
    ]
    if path_save:
        np.save(path_save, np.array(total_array, dtype=object))
    return total_array


def load_model_and_test_two_models(
    data_path,
    model_path_initialize,
    model_path_resample,
    n_steps: int,
    n_runs: int,
    algo_type,
    path_save=False,
    keep_traj=True,
):
    """Run oracle versions of MT and Walksat algorithm on a dataset to evaluate the performance.

    Here you have the additional freedom to use different models for initialization and resampling in the algorithms

    Args:
        data_path (str): path pointing to dataset used for evaluation
        model_path_initialize (str): path of the model or "uniform" for running the uniform version. This is used for initialization.
        model_path_resample (str): path of the model or "uniform" for running the uniform version. This is used for resampling.
        n_steps (int): number of steps taken by the algorithm
        n_runs (int): number of runs executed by the algorithm
        algo_type (str): either "moser" for MT algorithm or "probsat" for the oracle version of WalkSAT
        path_save (bool, optional): path where details of the experiment are saved. Defaults to False.
        keep_traj (bool, optional): decide whether you want to keep the trajectories. Defaults to True.

    Returns:
        list:       Returns the following elements. They are also saved at path_save.
                        [model_path_initialize, model_path_resample],
                        [model_details_list_i, model_details_list_r],
                        n_array,
                        alpha_array,
                        energies_array_mean,
                        energies_array_median,
                        total_steps,
    """
    energies_array_mean = []
    energies_array_median = []

    n_array = []
    alpha_array = []
    total_steps = list([])
    if model_path_initialize != "uniform":
        params_i, model_details_i = np.load(model_path_initialize, allow_pickle=True)
        (
            inv_temp_i,
            alpha_i,
            beta_i,
            gamma_i,
            mlp_layers_i,
            graph_representation_i,
            network_type_i,
            return_candidates_i,
        ) = model_details_i
        model_details_list_i = [
            inv_temp_i,
            alpha_i,
            beta_i,
            gamma_i,
            mlp_layers_i,
            graph_representation_i.__name__,
            network_type_i,
            return_candidates_i,
        ]
        include_constraint_graph_i = beta_i > 0
        sat_data = SATTrainingDataset(
            data_path,
            graph_representation_i,
            return_candidates=return_candidates_i,
            include_constraint_graph=include_constraint_graph_i,
        )
        network_definition_i = get_network_definition(
            network_type=network_type_i, graph_representation=graph_representation_i
        )
        network_definition_i = partial(network_definition_i, mlp_layers=mlp_layers_i)
        network_i = hk.without_apply_rng(hk.transform(network_definition_i))
    else:
        sat_data = SATTrainingDataset(
            data_path, LCG, return_candidates=False, include_constraint_graph=False
        )
        model_details_list_i = []
    if model_path_resample != "uniform":
        params_r, model_details_r = np.load(model_path_resample, allow_pickle=True)
        (
            inv_temp_r,
            alpha_r,
            beta_r,
            gamma_r,
            mlp_layers_r,
            graph_representation_r,
            network_type_r,
            return_candidates_r,
        ) = model_details_r
        model_details_list_r = [
            inv_temp_r,
            alpha_r,
            beta_r,
            gamma_r,
            mlp_layers_r,
            graph_representation_r.__name__,
            network_type_r,
            return_candidates_r,
        ]
        network_definition_r = get_network_definition(
            network_type=network_type_r, graph_representation=graph_representation_r
        )
        network_definition_r = partial(network_definition_r, mlp_layers=mlp_layers_r)
        network_r = hk.without_apply_rng(hk.transform(network_definition_r))
    else:
        model_details_list_r = []
    for idx in range(len(sat_data)):
        print("problem ", idx + 1, "of ", len(sat_data))
        problem_path = sat_data.instances[idx].name + ".cnf"
        problem = sat_data.get_unpadded_problem(idx)

        n_array.append(problem.params[0])
        alpha_array.append(problem.params[1] / problem.params[0])
        if model_path_initialize != "uniform":
            decoded_nodes = network_i.apply(params_i, problem.graph)  # type: ignore[attr-defined]
            n_variables = problem.params[0]
            model_probabilities_i = graph_representation_i.get_model_probabilities(
                decoded_nodes, n_variables
            )
            model_probabilities_i = model_probabilities_i.ravel()
        else:
            model_probabilities_i = np.ones(problem.params[0]) / 2
        if model_path_resample != "uniform":
            decoded_nodes = network_r.apply(params_r, problem.graph)  # type: ignore[attr-defined]
            n_variables = problem.params[0]
            model_probabilities_r = graph_representation_r.get_model_probabilities(
                decoded_nodes, n_variables
            )
            model_probabilities_r = model_probabilities_r.ravel()
        else:
            model_probabilities_r = np.ones(problem.params[0]) / 2
        single_traj_mean = []
        single_traj_median = []
        print("model initialize", model_path_initialize)
        print("model resample", model_path_resample)

        _, _, _, numstep, traj = moser_rust.run_sls_python(
            algo_type,
            problem_path,
            model_probabilities_i,
            model_probabilities_r,
            n_steps - 1,
            n_runs,
            SEED,
            keep_traj,
        )
        total_steps.append(numstep)
        if len(traj) != 0:
            traj = get_padded_trajs(traj, n_steps)
            single_traj_mean.append(np.mean(traj, axis=0) / problem.params[1])
            single_traj_median.append(np.median(traj, axis=0) / problem.params[1])
            energies_array_mean.append(
                np.pad(
                    np.array(single_traj_mean)[0],
                    (0, n_steps - len(single_traj_mean[0])),
                )
            )
            energies_array_median.append(
                np.pad(
                    np.array(single_traj_median)[0],
                    (0, n_steps - len(single_traj_median[0])),
                )
            )
    total_steps_array = np.asarray(total_steps)
    if energies_array_mean:
        energies_array_mean = np.mean(energies_array_mean, axis=0)
        energies_array_median = np.mean(energies_array_median, axis=0)
    total_array = [
        [model_path_initialize, model_path_resample],
        [model_details_list_i, model_details_list_r],
        n_array,
        alpha_array,
        energies_array_mean,
        energies_array_median,
        total_steps_array,
    ]
    if path_save:
        np.save(path_save, np.array(total_array, dtype=object))
    return total_array
