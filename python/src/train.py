import sys

sys.path.append("../../")

import collections

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from torch.utils import data
import matplotlib.pyplot as plt
import pandas as pd
import moser_rust
from data_utils import SATTrainingDataset, JraphDataLoader
from model import (
    network_definition_interaction,
    network_definition_GCN,
    get_model_probabilities,
)
from random_walk import moser_walk
import mlflow
from pathlib import Path
import tempfile
import joblib
from jraph._src import utils
import jraph
from jax.experimental.sparse import BCOO
from pysat.formula import CNF

NUM_EPOCHS = 20  # 10
f = 0.01
batch_size = 1
# path = "../Data/blocksworld"
path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/generateSAT/samples_medium"
N_STEPS_MOSER = 100
N_RUNS_MOSER = 2
SEED = 0

MODEL_REGISTRY = Path("experiment_tracking/experiments_storing")
EXPERIMENT_NAME = "mlflow-demo2"


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
):
    network_definition = network_definition_interaction
    sat_data = SATTrainingDataset(path)
    # print(sat_data[0][0].graph.edges)
    train_data, test_data = data.random_split(sat_data, [0.8, 0.2])
    train_eval_data, _ = data.random_split(train_data, [0.2, 0.8])

    train_loader = JraphDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = JraphDataLoader(test_data, batch_size=batch_size)
    train_eval_loader = JraphDataLoader(train_eval_data, batch_size=batch_size)

    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @jax.jit
    def update(params, opt_state, batch, f):
        # g = jax.grad(local_lovasz_loss)(params, batch)
        g = jax.grad(combined_loss)(params, batch, f)

        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    def local_lovasz_loss(params, batch, f=False):
        """
        This assumes that the output of the graph at this point is 2 dimensional
        """
        (mask, graph), _ = batch
        decoded_nodes = network.apply(params, graph)  # (B*N, 2)
        log_probs = jax.nn.log_softmax(
            decoded_nodes
        )  # the log probs for each node (variable and constraint) # (B*N, 2)
        e = len(graph.edges)
        # n = graph.n_node
        n = jnp.shape(decoded_nodes)[0]

        constraint_node_mask = jnp.array(jnp.logical_not(mask), dtype=int)

        # calculate the probability of a constraint being violated by summing the varaible node probs according to the violated string
        relevant_log_probs = jnp.sum(log_probs[graph.senders] * graph.edges, axis=1)
        # relevant_log_probs = log_probs[graph.senders][jnp.arange(int(n)).astype(int), graph.edges[:, 1].astype(int)]
        convolved_log_probs = utils.segment_sum(
            relevant_log_probs, graph.receivers, num_segments=n
        )
        lhs_values = convolved_log_probs * constraint_node_mask
        # print("lhs_values", lhs_values.shape)
        # calculate RHS of inequalities:
        rhs_sums = utils.segment_sum(
            data=log_probs[graph.receivers]
            * constraint_node_mask[graph.receivers][:, None],
            segment_ids=graph.senders,
            num_segments=n,
        )
        rhs_values = rhs_sums[:, 1] + log_probs[:, 0]
        rhs_values = rhs_values * constraint_node_mask
        # print("rhs_values", rhs_values.shape)
        """
        # First calculate the two hop edge information

        adjacency_matrix = BCOO(
            (
                jnp.ones(e),
                jnp.column_stack((graph.senders, graph.receivers)),
            ),
            shape=(n, n),
            unique_indices=True,
        )

        # two hop adjacency matrix with values indicating number of shared two hop paths.
        # adj_squared = jnp.matmul(adjacency_matrix, adjacency_matrix)

        adj_squared = jax.experimental.sparse.bcoo_multiply_sparse(
            adjacency_matrix, adjacency_matrix
        )
        induced_indices = adj_squared.indices[adj_squared.data != 0]
        constraint_senders = induced_indices[:, 0]
        constraint_receivers = induced_indices[:, 1]

        rhs_sums = utils.segment_sum(
            log_probs[constraint_senders], constraint_receivers, num_segments=n
        )
        rhs_values = rhs_sums[:, 1] + log_probs[:, 0]
        rhs_values = rhs_values * constraint_node_mask
        """
        # PAUL'S IDEA:
        """
        # mask for all possible two hop paths between constraint nodes
        shared_path_mask = jnp.tile(graph.senders, e) == jnp.repeat(graph.receivers, e)
        print("shared_path_mask", shared_path_mask.shape)
        # for each node, we sum the log probs for all shared incoming paths
        # TODO: Still have to deal with double counting here
        rhs_sums = utils.segment_sum(
            data=jnp.where(
                shared_path_mask,
                log_probs[jnp.tile(graph.receivers, e)][:, 1],
                0,
            ),
            segment_ids=jnp.repeat(graph.receivers, e),
            num_segments=n,
        )
        print("rhs_sums", rhs_sums.shape)
        rhs_values = rhs_sums[:, 1] + log_probs[:, 0]
        rhs_values = rhs_values * constraint_node_mask
        print(rhs_values.shape)
        """
        # IDEA MAX: no double-counting
        """
        edges = graph.edges[:, 0] - graph.edges[:, 1]
        adjacency_matrix = BCOO(
            (
                edges,
                jnp.column_stack((graph.receivers, graph.senders)),
            ),
            shape=(n, n),
            unique_indices=True,
        )
        adjacency_matrix = adjacency_matrix.todense()

        def get_neighborhood(matrix, conflicting_only=False):
            occurencies_matrix = abs(matrix)
            neighbors_indices_list = []

            for i in range(jnp.shape(occurencies_matrix)[1]):
                target = matrix[:, i][:, None]

                if conflicting_only == False:
                    a = jnp.where(
                        jnp.sum(occurencies_matrix[:, 0:i] * abs(target), axis=0) != 0,
                        1,
                        0,
                    )
                    b = jnp.where(
                        jnp.sum(
                            occurencies_matrix[
                                :, i + 1 : jnp.shape(occurencies_matrix)[1]
                            ]
                            * abs(target),
                            axis=0,
                        )
                        != 0,
                        1,
                        0,
                    )
                if conflicting_only == True:
                    a = jnp.where(
                        jnp.sum(
                            (
                                (occurencies_matrix[:, 0:i] * abs(target))
                                - (matrix[:, 0:i] * target)
                            )
                            / 2,
                            axis=0,
                        )
                        != 0,
                        1,
                        0,
                    )
                    b = jnp.where(
                        jnp.sum(
                            (
                                (
                                    occurencies_matrix[
                                        :, i + 1 : jnp.shape(occurencies_matrix)[1]
                                    ]
                                    * abs(target)
                                )
                                - (
                                    matrix[:, i + 1 : jnp.shape(occurencies_matrix)[1]]
                                    * target
                                )
                            )
                            / 2,
                            axis=0,
                        )
                        != 0,
                        1,
                        0,
                    )

                a = jnp.hstack((a, [0]))
                remainder = jnp.hstack((a, b))
                neighbors_indices = jnp.argwhere(remainder != 0).transpose()[0]
                # if len(neighbors_indices) != 0:
                neighbors_indices_list.append(neighbors_indices)
            return neighbors_indices_list

        neighbors_list = get_neighborhood(
            adjacency_matrix, conflicting_only=False
        )  # this could be pre-computed and loaded for every problem / graph...

        def get_rhs_for_neighborhood(neighbors):
            rhs_single = jnp.sum(
                jnp.array([log_probs[neighbor, 1] for neighbor in neighbors]), axis=0
            )
            return rhs_single

        rhs_values = jnp.array(get_rhs_for_neighborhood(neighbors_list[0]))[None]
        for i in range(1, len(neighbors_list)):
            a = jnp.array(get_rhs_for_neighborhood(neighbors_list[i]))[None]
            rhs_values = jnp.append(rhs_values, a)
        rhs_values += log_probs[:, 0]
        """
        # using the relative entropy as proxy for max relative entropy for sake of differentiability
        # (could move to 2-renyi divergence later or higher)
        # def RelativeEntropy(A, B):
        #    return jnp.sum(jnp.where(B != 0, A * jnp.log(A / B), 0))

        # TODO: probably we'll have to do some masking at this last stage
        # TODO: Dealing with batching

        # loss = RelativeEntropy(np.exp(lhs_values), np.exp(rhs_values))
        # loss = RelativeEntropy(lhs_values, rhs_values)
        difference = lhs_values - rhs_values
        loss = jnp.maximum(difference, np.zeros(len(rhs_values)))
        return jnp.sum(loss, axis=0)

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

    def combined_loss(params, batch, f: float):
        return 0.05 * prediction_loss(params, batch, f) + local_lovasz_loss(
            params, batch
        )

    print("Entering training loop")

    def evaluate(loader, loss):
        return np.mean([loss(params, b, f) for b in loader])

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
                problem_path,
                model_probabilities.ravel(),
                N_STEPS_MOSER,
                N_RUNS_MOSER,
                SEED,
            )
            _, m, _ = problem.params
            av_energies.append(np.mean(final_energies) / m)

        return np.mean(av_energies)

    test_eval = EvalResults("Test loss", [], True)
    train_eval = EvalResults("Train loss", [], True)
    test_moser_eval = EvalResults("Moser loss - test", [], False)
    train_moser_eval = EvalResults("Moser loss - train", [], False)
    test_eval_lll = EvalResults("Test loss LLL", [], True)
    train_eval_lll = EvalResults("Train loss LLL", [], True)
    test_eval_dm = EvalResults("Test loss Deepmind", [], True)
    train_eval_dm = EvalResults("Train loss Deepmind", [], True)
    eval_objects = [
        test_eval,
        train_eval,
        test_moser_eval,
        train_moser_eval,
        test_eval_lll,
        train_eval_lll,
        test_eval_dm,
        train_eval_dm,
    ]

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        for counter, batch in enumerate(train_loader):
            # print("batch_number", counter)
            params, opt_state = update(params, opt_state, batch, f)

        epoch_time = time.time() - start_time

        # print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))

        test_eval.results.append(evaluate(test_loader, combined_loss))
        train_eval.results.append(evaluate(train_eval_loader, combined_loss))
        test_eval_lll.results.append(evaluate(test_loader, local_lovasz_loss))
        train_eval_lll.results.append(evaluate(train_eval_loader, local_lovasz_loss))
        test_eval_dm.results.append(evaluate(test_loader, prediction_loss))
        train_eval_dm.results.append(evaluate(train_eval_loader, prediction_loss))
        test_moser_eval.results.append(evaluate_moser_rust(test_data))
        train_moser_eval.results.append(evaluate_moser_rust(train_eval_data))
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

"""
def experiment_tracking_train(
    MODEL_REGISTRY,
    EXPERIMENT_NAME,
    batch_size,
    f,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    path,
    img_path=False,
    model_path=False,
):
    Path(MODEL_REGISTRY).mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # train and evaluate
        artifacts = train(
            batch_size,
            f,
            NUM_EPOCHS,
            N_STEPS_MOSER,
            path,
            img_path,
            model_path,
            experiment_tracking=True,
        )
        # log key hyperparameters
        mlflow.log_params(
            {
                "f": f,
                "batch_size": batch_size,
                "NUM_EPOCHS": NUM_EPOCHS,
                "N_STEPS_MOSER": N_STEPS_MOSER,
            }
        )
        # log params which are a result of learning
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["params"], Path(dp, "params.pkl"))
            mlflow.log_artifact(dp)


if __name__ == "__main__":
    experiment_tracking_train(MODEL_REGISTRY, EXPERIMENT_NAME)
"""

if __name__ == "__main__":
    train(
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
