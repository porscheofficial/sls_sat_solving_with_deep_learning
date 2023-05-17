import collections
import numpy as np
import matplotlib.pyplot as plt
import moser_rust
from scipy.stats import entropy
import pandas as pd

from python.src.sat_representations import SATRepresentation

EvalResults = collections.namedtuple(
    "EvalResult", ("name", "results", "normalize", "loss_params", "rep", "loader")
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
    eval_total = EvalResults(
        text + " total loss", [], False, [f, alpha, beta, gamma], rep, loader
    )
    eval_dm = EvalResults(
        text + " loss Deepmind", [], False, [f, alpha, 0, 0], rep, loader
    )
    eval_lll = EvalResults(text + " loss LLL", [], False, [f, 0, beta, 0], rep, loader)
    eval_entropy = EvalResults(
        text + " loss entropy", [], False, [f, 0, 0, gamma], rep, loader
    )

    eval_objects_loss = [
        eval_total,
        eval_dm,
        eval_lll,
        eval_entropy,
    ]

    return eval_objects_loss


def update_eval_objects_loss(params, loss, eval_objects_loss):
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
    print(eval_objects_loss)
    return eval_objects_loss


def plot_accuracy_fig(*eval_results):
    ROLLING_WINDOW_SIZE = 10
    for eval_result in eval_results:
        results = np.array(eval_result.results)
        if eval_result.normalize:
            results /= np.max(results)
        plt.plot(
            # np.arange(0, NUM_EPOCHS - ROLLING_WINDOW_SIZE, 1),
            pd.Series(results).rolling(ROLLING_WINDOW_SIZE).mean(),
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
        print(np.round(model_probabilities, 4))
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
