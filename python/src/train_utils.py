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
    f: float,
    alpha: float,
    beta: float,
    gamma: float,
    rep: SATRepresentation,
    loader,
):
    """helper function to create loss function evaluation object

    Args:
        text (str): description of loss type as str (typically "test" and "train")
        f (float): inverse temperature
        alpha (float): used pre-factor for the Gibbs loss
        beta (float): used pre-factor for the LLL loss
        gamma (float): used pre-factor for the alternative LLL loss
        rep (SATRepresentation): SATRepresentation used here
        loader (@TODO: specify type): _description_

    Returns:
        EvalResults: initialised EvalResults object for loss functions
    """
    eval_total = EvalResults(
        text + " total loss", [], False, [f, alpha, beta, gamma], rep, loader
    )
    eval_dm = EvalResults(
        text + " loss Gibbs", [], False, [f, alpha, 0, 0], rep, loader
    )
    eval_lll = EvalResults(text + " loss LLL", [], False, [f, 0, beta, 0], rep, loader)
    eval_alt_lll = EvalResults(
        text + " loss alt_LLL", [], False, [f, 0, 0, gamma], rep, loader
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
    N_STEPS_MOSER: float,
    N_RUNS_MOSER: float,
    rep: SATRepresentation,
    data_subset,
    sat_data,
):
    """helper function to create a moser_loss evaluation object

    Args:
        text (str): description of loss type as str (typically "test" and "train")
        N_STEPS_MOSER (float): number of steps executed by MT algorithm for the loss
        N_RUNS_MOSER (float): number of runs executed by MT algorithm for the loss
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
        [N_STEPS_MOSER, N_RUNS_MOSER, "model"],
        rep,
        [data_subset, sat_data],
    )
    moser_uniform = EvalResults(
        text + " loss uniform Moser",
        [],
        False,
        [N_STEPS_MOSER, N_RUNS_MOSER, "uniform"],
        rep,
        [data_subset, sat_data],
    )

    eval_moser_loss = [moser_model, moser_uniform]

    return eval_moser_loss


def update_eval_objects_loss(params, loss, eval_objects_loss):
    """helper function to update loss function evaluation object. This means, use the current params of the net and evaluate the loss terms

    Args:
        params (@TODO: specify type): current params of the net
        loss (function): function we want to evaluate -> use the total loss function typically
        eval_objects_loss (EvalResults): current EvalResults object that should be updated using this function

    Returns:
        EvalResults: updated EvalResults loss function object
    """
    for i in range(len(eval_objects_loss)):
        eval_objects_loss[i].results.append(
            np.mean(
                [
                    loss(
                        params,
                        b,
                        eval_objects_loss[i].loss_params[0],
                        eval_objects_loss[i].loss_params[1],
                        eval_objects_loss[i].loss_params[2],
                        eval_objects_loss[i].loss_params[3],
                        eval_objects_loss[i].rep,
                    )
                    for b in eval_objects_loss[i].loader
                ]
            )
        )
    return eval_objects_loss


def update_eval_moser_loss(network, params, eval_moser_loss):
    """helper function to update moser loss evaluation object. This means, use the current params of the net and evaluate the Moser loss terms

    Args:
        network (@TODO: specify type): network definition
        params (@TODO: specify type): current params of the net
        eval_moser_loss (EvalResults): current EvalResults object that should be updated using this function

    Returns:
        EvalResults: updated EvalResults Moser loss object
    """
    for i in range(len(eval_moser_loss)):
        if (
            eval_moser_loss[i].loss_params[2] == "model"
            or len(eval_moser_loss[i].results) == 0
        ):
            eval_moser_loss[i].results.append(
                evaluate_moser_rust(
                    eval_moser_loss[i].loader[1],
                    network,
                    params,
                    eval_moser_loss[i].loader[0],
                    eval_moser_loss[i].rep,
                    mode_probabilities=eval_moser_loss[i].loss_params[2],
                    N_STEPS_MOSER=eval_moser_loss[i].loss_params[0],
                    N_RUNS_MOSER=eval_moser_loss[i].loss_params[1],
                    SEED=0,
                )[0]
            )
        elif eval_moser_loss[i].loss_params[2] == "uniform":
            eval_moser_loss[i].results.append(eval_moser_loss[i].results[0])
    return eval_moser_loss


def plot_accuracy_fig(*eval_results):
    """helper function to do a plot containing the losses as a function of the epochs."""
    ROLLING_WINDOW_SIZE = 1
    for eval_result in eval_results:
        results = np.array(eval_result.results)
        if eval_result.normalize:
            results /= np.max(results)
        plt.plot(
            # np.arange(0, NUM_EPOCHS - ROLLING_WINDOW_SIZE, 1),
            pd.Series(results)
            .rolling(np.min([ROLLING_WINDOW_SIZE, len(results)]))
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
    N_STEPS_MOSER=100,
    N_RUNS_MOSER=1,
    SEED=0,
):
    """Function to run MT algorithm in rust and get results. This is used above.

    Args:
        sat_data (@TODO: specify type): sat_data used as input problem
        network (@TODO: specify type): network definition
        params (@TODO: specify type): params of the net
        data_subset (@TODO: specify type): data_subset used for evaluation
        representation (SATRepresentation): SATRepresentation used here
        mode_probabilities (str, optional): either "model" or "uniform" -> "model" means we use the model for the oracle and "uniform" means we use a uniform oracle in the MT algorithm. Defaults to "model".
        N_STEPS_MOSER (int, optional): number of steps MT algorithm takes. Defaults to 100.
        N_RUNS_MOSER (int, optional): number of runs MT algorithm takes. Defaults to 1.
        SEED (int, optional): SEED used in MT algorithm. Defaults to 0.

    Returns:
        _type_: (np.mean(av_energies), np.mean(av_entropies)) -> mean energies = #violated clauses / #clause after N_STEPS_MOSER and mean entropy of the oracle probabilities
    """
    av_energies = []
    av_entropies = []

    for idx in data_subset.indices:
        problem_path = sat_data.instances[idx].name + ".cnf"
        problem = sat_data.get_unpadded_problem(idx)
        n, _, _ = problem.params
        if mode_probabilities == "uniform":
            model_probabilities = np.ones(n) / 2
        elif mode_probabilities == "model":
            decoded_nodes = network.apply(params, problem.graph)
            model_probabilities = representation.get_model_probabilities(
                decoded_nodes, n
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

        _, _, final_energies, _, _, _ = moser_rust.run_sls_python(
            "moser",
            problem_path,
            model_probabilities,
            N_STEPS_MOSER,
            N_RUNS_MOSER,
            SEED,
            return_trajectories=False,
        )
        _, m, _ = problem.params
        av_energies.append(np.mean(final_energies) / m)
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
