import sys

sys.path.append("../../")

import collections

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import optax
import time
from torch.utils import data
import matplotlib.pyplot as plt

import moser_rust
from data_utils import SATTrainingDataset, JraphDataLoader
from model import network_definition, get_model_probabilities
from random_walk import moser_walk
import mlflow
from pathlib import Path
import tempfile
import joblib

from os import walk


NUM_EPOCHS = 5000  # 10
f = 0.01
batch_size = 2
path = "../data/blocksworld_subset"
filenames = next(walk(path), (None, None, []))[2]
print(filenames)
N_STEPS_MOSER = 10000
N_RUNS_MOSER = 2


#  AUXILIARY METHODS


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)


vmap_one_hot = jax.vmap(one_hot, in_axes=(0, None), out_axes=0)


def compute_log_probs(decoded_nodes, mask, candidate):
    a = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
    return candidate * a


vmap_compute_log_probs = jax.vmap(
    compute_log_probs, in_axes=(None, None, 1), out_axes=1
)


def evaluate_on_moser(
    network,
    params,
    problem,
    n_steps,
    keep_trajectory=False,
):
    model_probabilities = get_model_probabilities(network, params, problem)
    output, energy, counter = moser_walk(
        model_probabilities, problem, n_steps, seed=0, keep_trajectory=keep_trajectory
    )
    n, _, _ = problem.params
    return np.min(energy) / n


def get_running_mean(arr, window_size):
    numbers_series = pd.Series(arr)
    windows = numbers_series.rolling(window_size)
    moving_averages = windows.mean()
    moving_averages_list = moving_averages.tolist()
    final_list = moving_averages_list[window_size - 1 :]
    return final_list


def plot_accuracy_fig(*eval_results):
    for eval_result in eval_results:

        plt.plot(
            np.arange(0, NUM_EPOCHS + 1, 1),
            np.array(eval_result.results),
            "o--",
            label=eval_result.name,
            alpha=0.05,
        )
        window_size = 40
        running_mean = get_running_mean(eval_result.results, window_size)
        plt.plot(
            np.arange(
                int(window_size / 2), len(running_mean) + int(window_size / 2), 1
            ),
            np.array(running_mean),
            "-",
            label=eval_result.name + "RM",
            alpha=0.9,
        )
    plt.xlabel("epoch")
    plt.ylabel("accuracy of model / loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


EvalResults = collections.namedtuple("EvalResult", ("name", "results"))


def train2(
    batch_size,
    f,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    N_RUNS_MOSER,
    path,
    img_path=False,
    model_path=False,
    experiment_tracking=False,
):
    sat_data = SATTrainingDataset(path)

    train_data, test_data = data.random_split(sat_data, [1, 0])
    # print(train_data)
    train_eval_data, _ = data.random_split(train_data, [0.2, 0.8])

    train_loader = JraphDataLoader(train_data, batch_size=batch_size, shuffle=True)
    # test_loader = JraphDataLoader(test_data, batch_size=batch_size)
    # train_eval_loader = JraphDataLoader(train_eval_data, batch_size=batch_size)

    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def update(params, opt_state, batch, f):

        g = jax.grad(prediction_loss)(params, batch, f)

        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    @jax.jit
    def prediction_loss(params, batch, f: float):
        (mask, graph), (candidates, energies) = batch
        decoded_nodes = network.apply(params, graph)  # (B*N, 2)
        candidates = vmap_one_hot(candidates, 2)  # (B*N, K, 2))
        log_prob = vmap_compute_log_probs(
            decoded_nodes, mask, candidates
        )  # (B*N, K, 2)
        weights = jax.nn.softmax(-f * energies)  # (B*N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(mask)  # ()
        return loss

    print("Entering training loop")

    def evaluate(loader):
        return np.mean([prediction_loss(params, b, f) for b in loader])

    def evaluate_moser_jax(data_subset):
        return np.mean(
            [
                evaluate_on_moser(
                    network, params, sat_data.get_unpadded_problem(i), N_STEPS_MOSER
                )
                for i in data_subset.indices
            ]
        )

    def evaluate_moser_rust(data_subset):

        av_energies = []

        for idx in data_subset.indices:
            problem_path = sat_data.instances[idx].name + ".cnf"
            problem = sat_data.get_unpadded_problem(idx)
            model_probabilities = get_model_probabilities(network, params, problem)
            _, _, final_energies = moser_rust.run_moser_python(
                problem_path, model_probabilities.ravel(), N_STEPS_MOSER, N_RUNS_MOSER
            )
            _, m, _ = problem.params
            av_energies.append(np.mean(final_energies) / m)

        return np.mean(av_energies)

    # test_eval = EvalResults("Test loss", [evaluate(test_loader)])
    train_eval = EvalResults("Train loss", [evaluate(train_loader)])
    # test_moser_eval = EvalResults("Moser loss (test)", [evaluate_moser_rust(test_data)])
    train_moser_eval_R = EvalResults(
        "Moser loss (train) RUST", [evaluate_moser_rust(train_data)]
    )
    train_moser_eval_J = EvalResults(
        "Moser loss (train) JAX", [evaluate_moser_jax(train_data)]
    )
    # eval_objects = [test_eval, train_eval, test_moser_eval, train_moser_eval]
    eval_objects = [train_eval, train_moser_eval_R, train_moser_eval_J]

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for counter, batch in enumerate(train_loader):
            # print("batch_number", counter)
            params, opt_state = update(params, opt_state, batch, f)

        epoch_time = time.time() - start_time

        # print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        # test_eval.results.append(evaluate(test_loader))
        train_eval.results.append(evaluate(train_loader))
        # test_moser_eval.results.append(evaluate_moser_rust(test_data))
        train_moser_eval_R.results.append(evaluate_moser_rust(train_data))
        train_moser_eval_J.results.append(evaluate_moser_jax(train_data))
        loss_str = "Epoch {} in {:0.2f} sec".format(epoch, epoch_time) + ";  "
        for eval_result in eval_objects:
            # print(f"{eval_result.name}: {eval_result.results[-1]}")
            loss_str = (
                loss_str
                + f"{eval_result.name}: {np.round(eval_result.results[-1],4)}"
                + "; "
            )
            if experiment_tracking == True:
                mlflow.log_metric(eval_result.name, eval_result.results[-1], step=epoch)
        print(loss_str)
    if img_path:
        plot_accuracy_fig(*eval_objects)
        if img_path == "show":
            plt.show()
        else:
            plt.savefig(img_path + "accuracy.jpg", dpi=300, format="jpg")

    if model_path:
        model_params = [params, batch_size, f, NUM_EPOCHS]
        np.save(model_path, [model_params, *eval_objects])

    return {
        "params": params,
    }


if __name__ == "__main__":
    train2(
        batch_size,
        f,
        NUM_EPOCHS,
        N_STEPS_MOSER,
        N_RUNS_MOSER,
        path,
        img_path="show",
        model_path=False,
        experiment_tracking=False,
    )
