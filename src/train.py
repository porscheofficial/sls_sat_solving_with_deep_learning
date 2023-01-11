import collections

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from torch.utils import data
import matplotlib.pyplot as plt

from data_utils import SATTrainingDataset, JraphDataLoader
from model import network_definition, get_model_probabilities
from random_walk import moser_walk
import mlflow
from pathlib import Path
import tempfile
import joblib

NUM_EPOCHS = 2  # 10
f = 0.1
batch_size = 2
path = "../Data/blocksworld"
N_STEPS_MOSER = 1000

MODEL_REGISTRY = Path("experiments")
Path(MODEL_REGISTRY).mkdir(exist_ok=True)  # create experiments dir
mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))

EXPERIMENT_NAME = "mlflow-demo"
# EXPERIMENT_ID = mlflow.create_experiment(EXPERIMENT_NAME)
EXPERIMENT_ID = mlflow.set_experiment(EXPERIMENT_NAME)


# "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/ml_based_sat_solver/BroadcastTestSet_subset"
# img_path = (
#     "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/"
# )
# model_path = (
#     "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/"
# )


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


def plot_accuracy_fig(*eval_results):
    for eval_result in eval_results:
        plt.plot(
            np.arange(0, NUM_EPOCHS + 1, 1),
            np.array(eval_result.results),
            "o--",
            label=eval_result.name,
            alpha=0.4,
        )
    plt.xlabel("epoch")
    plt.ylabel("accuracy of model / loss")
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


EvalResults = collections.namedtuple("EvalResult", ("name", "results"))


def train(
    batch_size,
    f,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    path,
    img_path=False,
    model_path=False,
):
    sat_data = SATTrainingDataset(path)

    train_data, test_data = data.random_split(sat_data, [0.8, 0.2])
    train_eval_data, _ = data.random_split(sat_data, [0.2, 0.8])

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

    evaluate = lambda loader: np.mean([prediction_loss(params, b, f) for b in loader])
    evaluate_moser = lambda data_subset: np.mean(
        [
            evaluate_on_moser(
                network, params, sat_data.get_unpadded_problem(i), N_STEPS_MOSER
            )
            for i in data_subset.indices
        ]
    )

    test_eval = EvalResults("Test loss", [evaluate(test_loader)])
    train_eval = EvalResults("Train loss", [evaluate(train_eval_loader)])
    test_moser_eval = EvalResults("Moser loss", [evaluate_moser(test_data)])
    eval_objects = [test_eval, train_eval, test_moser_eval]

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for counter, batch in enumerate(train_loader):
            print("batch_number", counter)
            params, opt_state = update(params, opt_state, batch, f)

        epoch_time = time.time() - start_time

        print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        test_eval.results.append(evaluate(test_loader))
        train_eval.results.append(evaluate(train_eval_loader))
        test_moser_eval.results.append(evaluate_moser(test_data))

        for eval_result in eval_objects:
            print(f"{eval_result.name}: {eval_result.results[-1]}")
            mlflow.log_metric(eval_result.name, eval_result.results[-1], step=epoch)

    if img_path:
        plot_accuracy_fig(*eval_objects)
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


if __name__ == "__main__":
    with mlflow.start_run():
        # train and evaluate
        artifacts = train(batch_size, f, NUM_EPOCHS, N_STEPS_MOSER, path)
        # log key hyperparameters
        mlflow.log_params(
            {
                "f": f,
                "batch_size": batch_size,
                "NUM_EPOCHS": NUM_EPOCHS,
                "N_STEPS_MOSER": N_STEPS_MOSER,
            }
        )
        # log params after learning
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["params"], Path(dp, "params.pkl"))
            # save_dict(artifacts["params"], Path(dp, "params.json"))
            mlflow.log_artifact(dp)
