"""Utility functions used in train.py."""
import collections
import numpy as np
import matplotlib.pyplot as plt
import moser_rust
from scipy.stats import entropy
import pandas as pd

from python.src.sat_representations import SATRepresentation

EvalResults = collections.namedtuple(
    "EvalResults", ("name", "results", "normalize", "loss_params", "rep", "loader")
)


def initiate_eval_objects_loss(
    text: str,
    inv_temp: float,
    alpha: float,
    beta: float,
    gamma: float,
    rep: SATRepresentation,
    loader,
):
    """Create loss function evaluation object.

    Args:
        text (str): description of loss type as str (typically "test" and "train")
        inv_temp (float): inverse temperature
        alpha (float): used pre-factor for the Gibbs loss
        beta (float): used pre-factor for the LLL loss
        gamma (float): used pre-factor for the alternative LLL loss
        rep (SATRepresentation): SATRepresentation used here
        loader (@TODO: specify type): _description_

    Returns:
        EvalResults: initialised EvalResults object for loss functions
    """
    eval_total = EvalResults(
        text + " total loss", [], False, [inv_temp, alpha, beta, gamma], rep, loader
    )
    eval_dm = EvalResults(
        text + " loss Gibbs", [], False, [inv_temp, alpha, 0, 0], rep, loader
    )
    eval_lll = EvalResults(
        text + " loss LLL", [], False, [inv_temp, 0, beta, 0], rep, loader
    )
    eval_alt_lll = EvalResults(
        text + " loss alt_LLL", [], False, [inv_temp, 0, 0, gamma], rep, loader
    )

    eval_objects_loss = [
        eval_total,
        eval_dm,
        eval_lll,
        # eval_entropy,
        eval_alt_lll,
    ]

    return eval_objects_loss


def initiate_eval_moser_loss(
    text: str,
    n_steps_moser: float,
    n_runs_moser: float,
    rep: SATRepresentation,
    data_subset,
    sat_data,
):
    """Create a moser_loss evaluation object.

    Args:
        text (str): description of loss type as str (typically "test" and "train")
        n_steps_moser (float): number of steps executed by MT algorithm for the loss
        n_runs_moser (float): number of runs executed by MT algorithm for the loss
        rep (SATRepresentation): SATRepresentation chosen here
        data_subset (@TODO: specify type): subset of dataset we want to look at
        sat_data (@TODO: specify type): full sat_data

    Returns:
        EvalResults: initialised EvalResults object for Moser loss
    """
    moser_model = EvalResults(
        text + " loss model Moser",
        [],
        False,
        [n_steps_moser, n_runs_moser, "model"],
        rep,
        [data_subset, sat_data],
    )
    moser_uniform = EvalResults(
        text + " loss uniform Moser",
        [],
        False,
        [n_steps_moser, n_runs_moser, "uniform"],
        rep,
        [data_subset, sat_data],
    )

    eval_moser_loss = [moser_model, moser_uniform]

    return eval_moser_loss


def update_eval_objects_loss(params, loss, eval_objects_loss):
    """Update loss function evaluation object. This means, use the current params of the net and evaluate the loss terms.

    Args:
        params (@TODO: specify type): current params of the net
        loss (function): function we want to evaluate -> use the total loss function typically
        eval_objects_loss (EvalResults): current EvalResults object that should be updated using this function

    Returns:
        EvalResults: updated EvalResults loss function object
    """
    for _, eval_object in enumerate(eval_objects_loss):
        eval_object.results.append(
            np.mean(
                [
                    loss(
                        params,
                        b,
                        eval_object.loss_params[0],
                        eval_object.loss_params[1],
                        eval_object.loss_params[2],
                        eval_object.loss_params[3],
                        eval_object.rep,
                    )
                    for b in eval_object.loader
                ]
            )
        )
    return eval_objects_loss


def initiate_eval_objetcts_train_test(
    inv_temp, alpha, beta, gamma, graph_representation, test_loader, train_loader
):
    """Initiate eval_objects both for train and test data.

    Args:
        inv_temp (float): inverse temperature used in Gibbs Loss
        alpha (float): prefactor used for Gibbs loss
        beta (float): prefactor used for LLL loss
        gamma (float): prefactor used for alternative LLL loss
        graph_representation (SATRepresentation): SATRepresentation used. Either LCG or VCG. Note: here you have to use the SATRepresentation!!!
        test_loader (@TODO: specify type): JraphDataLoader object of test_data
        train_loader (@TODO: specify type): JraphDataLoader object of train_data

    Returns:
        eval_objects both for train and test data
    """
    test_eval_objects_loss = initiate_eval_objects_loss(
        "test", inv_temp, alpha, beta, gamma, graph_representation, test_loader
    )
    train_eval_objects_loss = initiate_eval_objects_loss(
        "train", inv_temp, alpha, beta, gamma, graph_representation, train_loader
    )
    eval_objects_loss = test_eval_objects_loss + train_eval_objects_loss
    return eval_objects_loss


def update_eval_moser_loss(network, params, eval_moser_loss):
    """Update moser loss evaluation object. This means, use the current params of the net and evaluate the Moser loss terms.

    Args:
        network (@TODO: specify type): network definition
        params (@TODO: specify type): current params of the net
        eval_moser_loss (EvalResults): current EvalResults object that should be updated using this function

    Returns:
        EvalResults: updated EvalResults Moser loss object
    """
    for i, eval_moser in enumerate(eval_moser_loss):
        if eval_moser.loss_params[2] == "model" or len(eval_moser.results) == 0:
            eval_moser.results.append(
                evaluate_moser_rust(
                    eval_moser.loader[1],
                    network,
                    params,
                    eval_moser.loader[0],
                    eval_moser.rep,
                    mode_probabilities=eval_moser.loss_params[2],
                    n_steps_moser=eval_moser.loss_params[0],
                    n_runs_moser=eval_moser.loss_params[1],
                    seed=0,
                )[0]
            )
        elif eval_moser_loss[i].loss_params[2] == "uniform":
            eval_moser_loss[i].results.append(eval_moser_loss[i].results[0])
    return eval_moser_loss


def initiate_eval_moser_train_test(
    n_steps_moser, n_runs_moser, graph_representation, test_data, train_data, sat_data
):
    """Initiate eval_moser both for train and test data.

    Args:
        n_steps_moser (int): number of steps used in MT algorithm statistics
        n_runs_moser (int): number of runs used in MT algorithm statistics
        graph_representation (SATRepresentation): SATRepresentation used. Either LCG or VCG. Note: here you have to use the SATRepresentation!!!
        test_data (@TODO: specify type): test_data = subset of sat_data used for testing
        train_data (@TODO: specify type): train_data = subset of sat_data used for training
        sat_data (@TODO: specify type): sat dataset

    Returns:
        eval_moser object both for training and testing
    """
    test_eval_moser_loss = initiate_eval_moser_loss(
        "test", n_steps_moser, n_runs_moser, graph_representation, test_data, sat_data
    )
    train_eval_moser_loss = initiate_eval_moser_loss(
        "train",
        n_steps_moser,
        n_runs_moser,
        graph_representation,
        train_data,
        sat_data,
    )
    eval_moser_loss = test_eval_moser_loss + train_eval_moser_loss
    return eval_moser_loss


def plot_accuracy_fig(*eval_results):
    """Do a plot containing the losses as a function of the epochs."""
    rolling_window_size = 1
    for eval_result in eval_results:
        results = np.array(eval_result.results)
        if eval_result.normalize:
            results /= np.max(results)
        plt.plot(
            # np.arange(0, NUM_EPOCHS - ROLLING_WINDOW_SIZE, 1),
            pd.Series(results)
            .rolling(np.min([rolling_window_size, len(results)]))
            .mean(),
            "o--",
            label=eval_result.name,
            alpha=0.4,
        )


## Evaluation Functions
def evaluate_moser_rust(
    sat_data,
    network,
    params,
    data_subset,
    representation: SATRepresentation,
    mode_probabilities="model",
    n_steps_moser=100,
    n_runs_moser=1,
    seed=0,
):
    """Run MT algorithm in rust and get results. This is used above.

    Args:
        sat_data (@TODO: specify type): sat_data used as input problem
        network (@TODO: specify type): network definition
        params (@TODO: specify type): params of the net
        data_subset (@TODO: specify type): data_subset used for evaluation
        representation (SATRepresentation): SATRepresentation used here
        mode_probabilities (str, optional): either "model" or "uniform" -> "model" means we use the model for the oracle and "uniform" means we use a uniform oracle in the MT algorithm. Defaults to "model".
        n_steps_moser (int, optional): number of steps MT algorithm takes. Defaults to 100.
        n_runs_moser (int, optional): number of runs MT algorithm takes. Defaults to 1.
        seed (int, optional): SEED used in MT algorithm. Defaults to 0.

    Returns:
        (@TODO:specify type): (np.mean(av_energies), np.mean(av_entropies)) -> mean energies = #violated clauses / #clause after N_STEPS_MOSER and mean entropy of the oracle probabilities
    """
    av_energies = []
    av_entropies = []

    for idx in data_subset.indices:
        problem_path = sat_data.instances[idx].name + ".cnf"
        problem = sat_data.get_unpadded_problem(idx)
        n_variables, n_clauses, _ = problem.params
        if mode_probabilities == "uniform":
            model_probabilities = np.ones(n_variables) / 2
        elif mode_probabilities == "model":
            decoded_nodes = network.apply(params, problem.graph)
            model_probabilities = representation.get_model_probabilities(
                decoded_nodes, n_variables
            )
        else:
            print("not valid argument for mode_probabilities")
        model_probabilities = model_probabilities.ravel()
        # print(np.round(model_probabilities, 4))
        print(
            np.max(model_probabilities),
            np.min(model_probabilities),
            model_probabilities.shape,
        )

        _, _, final_energies, _, _ = moser_rust.run_sls_python(
            "moser",
            problem_path,
            model_probabilities,
            model_probabilities,
            n_steps_moser,
            n_runs_moser,
            seed,
            return_trajectories=False,
        )
        av_energies.append(np.mean(final_energies) / n_clauses)
        prob = np.vstack(
            (
                np.ones(model_probabilities.shape[0]) - model_probabilities,
                model_probabilities,
            )
        )
        entropies = [
            entropy(prob[:, i], qk=None, base=2, axis=0)
            for i in range(np.shape(prob)[1])
        ]
        av_entropies.append(np.mean(entropies))

    return (np.mean(av_energies), np.mean(av_entropies))
