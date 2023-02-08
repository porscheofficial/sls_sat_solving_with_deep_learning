## DEFINING A TRAINING PIPELINE

# import sys
# sys.path.append('../../../')
# print(sys.path)

import collections

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from torch.utils import data
from torch import Generator
import matplotlib.pyplot as plt
import pandas as pd
import moser_rust
from data_utils import SATTrainingDataset, JraphDataLoader
from model import (
    network_definition_interaction,
    network_definition_interaction_single_output,
    network_definition_GCN,
    network_definition_GCN_single_output,
    get_model_probabilities,
)
from random_walk import moser_walk
import mlflow
from pathlib import Path
import tempfile
import joblib

NUM_EPOCHS = 20  # 10
f = 0.01
batch_size = 4
# path = "../Data/blocksworld"
path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/generateSAT/samples_medium/"
N_STEPS_MOSER = 1000
N_RUNS_MOSER = 2
SEED = 0
network_definition = network_definition_GCN_single_output

MODEL_REGISTRY = Path("mlrun")
# EXPERIMENT_NAME = "mlflow-blocksat_Interaction_LCG"
EXPERIMENT_NAME = "mlflow-random_3SAT-medium-Interaction-LCG_subset"


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


"""
def compute_log_probs(decoded_nodes, mask, candidate):
    a = jax.nn.log_softmax(decoded_nodes) * mask[:, None]
    return candidate * a


vmap_compute_log_probs = jax.vmap(
    compute_log_probs, in_axes=(None, None, 1), out_axes=1
)
"""


def evaluate_on_moser(
    network,
    params,
    problem,
    n_steps,
    keep_trajectory=False,
):
    model_probabilities = get_model_probabilities(network, params, problem)
    _, energy, _ = moser_walk(
        model_probabilities, problem, n_steps, seed=0, keep_trajectory=keep_trajectory
    )
    _, m, _ = problem.params
    return np.min(energy) / m


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
    plt.xlabel("epoch")
    plt.ylabel("accuracy of model / loss")
    plt.yscale("log")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


EvalResults = collections.namedtuple("EvalResult", ("name", "results", "normalize"))


def train(
    batch_size,
    f,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    N_RUNS_MOSER,
    path,
    img_path=False,
    model_path=False,
    experiment_tracking=False,
    network_definition=network_definition_interaction,
):
    sat_data = SATTrainingDataset(path)
    train_data, test_data = data.random_split(
        sat_data, [0.8, 0.2], generator=Generator().manual_seed(0)
    )
    train_eval_data, _ = data.random_split(
        train_data, [0.2, 0.8], generator=Generator().manual_seed(0)
    )

    train_loader = JraphDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = JraphDataLoader(test_data, batch_size=batch_size)
    train_eval_loader = JraphDataLoader(train_eval_data, batch_size=batch_size)

    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def update(params, opt_state, batch, f):

        g = jax.grad(prediction_loss)(params, batch, f)

        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    """
    def prediction_loss(params, batch, f: float):
        (mask, graph), (candidates, energies) = batch
        decoded_nodes = network.apply(params, graph)  # (B*N, 2)
        candidates = vmap_one_hot(candidates, 2)  # (B*N, K, 2))
        log_prob = vmap_compute_log_probs(
            decoded_nodes, mask, candidates
        )  # (B*N, K, 2)
        weights = jax.nn.softmax(-f * energies)  # (B*N, K)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1))  # ()
        # jax.debug.print("🤯 {x} 🤯", x=loss)
        return loss
    """

    def prediction_loss(params, batch, f: float):
        (mask, graph), (candidates, energies) = batch
        decoded_nodes = network.apply(params, graph)  # (B*2*N, 1)
        if np.shape(decoded_nodes)[0] % 2 == 1:
            # print(np.shape(decoded_nodes))
            # print(type(decoded_nodes))
            decoded_nodes = jnp.vstack((jnp.asarray(decoded_nodes), [[0]]))
            # decoded_nodes = jnp.concatenate((jnp.asarray(decoded_nodes), [[0]]))
            # print(np.shape(decoded_nodes))
            conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
            padded_conc_decoded_nodes = jnp.concatenate(
                (
                    jnp.asarray(conc_decoded_nodes),
                    np.zeros(np.shape(conc_decoded_nodes)),
                )
            )[:-1, :]
            # padded_conc_decoded_nodes = jnp.concatenate(
            #    (jnp.asarray(conc_decoded_nodes), jnp.asarray(conc_decoded_nodes))
            # )[:-1, :]
        else:
            conc_decoded_nodes = jnp.reshape(decoded_nodes, (-1, 2))
            padded_conc_decoded_nodes = jnp.concatenate(
                (conc_decoded_nodes, conc_decoded_nodes)
            )
        candidates = vmap_one_hot(candidates, 2)
        log_prob = vmap_compute_log_probs(padded_conc_decoded_nodes, mask, candidates)
        weights = jax.nn.softmax(-f * energies)
        loss = -jnp.sum(weights * jnp.sum(log_prob, axis=-1)) / jnp.sum(mask)  # ()
        # jax.debug.print("🤯 {x} 🤯", x=loss)
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

    def evaluate_moser_rust(data_subset, mode_probabilities="model"):

        av_energies = []

        for idx in data_subset.indices:
            problem_path = sat_data.instances[idx].name + ".cnf"
            problem = sat_data.get_unpadded_problem(idx)
            if mode_probabilities == "model":
                model_probabilities = get_model_probabilities(network, params, problem)
            elif mode_probabilities == "uniform":
                n, _, _ = problem.params
                model_probabilities = np.ones(n) / 2
            _, _, final_energies = moser_rust.run_moser_python(
                problem_path,
                model_probabilities.ravel(),
                N_STEPS_MOSER,
                N_RUNS_MOSER,
                SEED,
            )
            _, m, _ = problem.params
            av_energies.append(np.mean(final_energies) / m)

        return np.mean(av_energies)

    moser_baseline_test = evaluate_moser_rust(test_data, mode_probabilities="uniform")
    moser_baseline_train = evaluate_moser_rust(
        train_eval_data, mode_probabilities="uniform"
    )

    test_eval = EvalResults("Test loss", [], True)
    train_eval = EvalResults("Train loss", [], True)
    test_moser_eval = EvalResults("Moser loss - test", [], False)
    train_moser_eval = EvalResults("Moser loss - train", [], False)
    test_baseline_moser_eval = EvalResults("uniform Moser loss - test", [], False)
    train_baseline_moser_eval = EvalResults("uniform Moser loss - train", [], False)
    eval_objects = [
        test_eval,
        train_eval,
        test_moser_eval,
        train_moser_eval,
        test_baseline_moser_eval,
        train_baseline_moser_eval,
    ]

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for counter, batch in enumerate(train_loader):
            # print("batch_number", counter)
            params, opt_state = update(params, opt_state, batch, f)

        epoch_time = time.time() - start_time

        # print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        test_eval.results.append(evaluate(test_loader))
        train_eval.results.append(evaluate(train_eval_loader))
        test_moser_eval.results.append(evaluate_moser_rust(test_data))
        train_moser_eval.results.append(evaluate_moser_rust(train_eval_data))
        train_baseline_moser_eval.results.append(moser_baseline_train)
        test_baseline_moser_eval.results.append(moser_baseline_test)
        loss_str = "Epoch {} in {:0.2f} sec".format(epoch, epoch_time) + ";  "
        for eval_result in eval_objects:
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


# def save_dict(d, filepath):
#    """Save dict to a json file."""
#    with open(filepath, "w") as fp:
#        json.dump(d, indent=2, sort_keys=False, fp=fp)


def experiment_tracking_train(
    MODEL_REGISTRY,
    EXPERIMENT_NAME,
    batch_size,
    f,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    N_RUNS_MOSER,
    path,
    img_path=False,
    model_path=False,
    network_definition=network_definition_interaction,
):
    Path(MODEL_REGISTRY).mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # log key hyperparameters
        mlflow.log_params(
            {
                "f": f,
                "batch_size": batch_size,
                "NUM_EPOCHS": NUM_EPOCHS,
                "N_STEPS_MOSER": N_STEPS_MOSER,
                "N_RUNS_MOSER": N_RUNS_MOSER,
                "network_definition": network_definition.__name__,
                "path_dataset": path,
            }
        )
        # train and evaluate
        artifacts = train(
            batch_size,
            f,
            NUM_EPOCHS,
            N_STEPS_MOSER,
            N_RUNS_MOSER,
            path,
            img_path=img_path,
            model_path=model_path,
            experiment_tracking=True,
            network_definition=network_definition,
        )
        # log params which are a result of learning
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["params"], Path(dp, "params.pkl"))
            mlflow.log_artifact(dp)


if __name__ == "__main__":
    experiment_tracking_train(
        MODEL_REGISTRY,
        EXPERIMENT_NAME,
        batch_size,
        f,
        NUM_EPOCHS,
        N_STEPS_MOSER,
        N_RUNS_MOSER,
        path,
        img_path="show",
        model_path=False,
        network_definition=network_definition,
    )
