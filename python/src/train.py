from functools import partial
import sys
import time
from pathlib import Path
import tempfile
import os
import joblib
import mlflow
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
from torch import Generator
from torch.utils import data
import matplotlib.pyplot as plt
from jsonargparse import CLI


sys.path.append("../../")

from python.src.data_utils import SATTrainingDataset, JraphDataLoader
from python.src.sat_representations import VCG, LCG, SATRepresentation
from python.src.model import get_network_definition
from python.src.train_utils import (
    plot_accuracy_fig,
    initiate_eval_objects_loss,
    update_eval_objects_loss,
    initiate_eval_moser_loss,
    update_eval_moser_loss,
)


def train(
    batch_size,
    inv_temp,
    alpha,
    beta,
    gamma,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    N_RUNS_MOSER,
    path,
    graph_representation: SATRepresentation,
    network_type,
    mlp_layers,
    img_path=False,
    model_path=False,
    experiment_tracking=False,
    return_candidates=False,
):
    include_constraint_graph = (
        beta > 0
    )  # we calculate the constraint graphs only if we use it to calculate the llloss

    sat_data = SATTrainingDataset(
        path,
        graph_representation,
        return_candidates=return_candidates,
        include_constraint_graph=include_constraint_graph,
    )
    train_data, test_data = data.random_split(
        sat_data, [0.8, 0.2], generator=Generator().manual_seed(0)
    )
    train_eval_data, _ = data.random_split(
        train_data, [0.2, 0.8], generator=Generator().manual_seed(0)
    )
    t1 = time.time()
    train_loader = JraphDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = JraphDataLoader(test_data, batch_size=batch_size)
    train_eval_loader = JraphDataLoader(train_eval_data, batch_size=batch_size)

    network_definition = get_network_definition(
        network_type=network_type, graph_representation=graph_representation
    )
    network_definition = partial(network_definition, mlp_layers=mlp_layers)
    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
    def total_loss(
        params,
        batch,
        inv_temp: float,
        alpha: float,
        beta: float,
        gamma: float,
        rep: SATRepresentation,
    ):
        (mask, graph, constraint_graph, constraint_mask), (candidates, energies) = batch
        decoded_nodes = network.apply(params, graph)
        prediction_loss = (
            alpha
            * rep.prediction_loss(decoded_nodes, mask, candidates, energies, inv_temp)
            if alpha > 0
            else 0.0
        )
        local_lovasz_loss = (
            beta
            * rep.local_lovasz_loss(
                decoded_nodes, mask, graph, constraint_graph, constraint_mask
            )
            if beta > 0
            else 0.0
        )
        entropy_loss = (
            gamma * rep.entropy_loss(decoded_nodes, mask) if gamma > 0 else 0.0
        )
        return prediction_loss + local_lovasz_loss + entropy_loss

    @jax.jit
    def update(params, batch, opt_state):
        g = jax.grad(total_loss)(
            params, batch, inv_temp, alpha, beta, gamma, graph_representation
        )
        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    print("Entering training loop")

    test_eval_objects_loss = initiate_eval_objects_loss(
        "test", inv_temp, alpha, beta, gamma, graph_representation, test_loader
    )
    train_eval_objects_loss = initiate_eval_objects_loss(
        "train", inv_temp, alpha, beta, gamma, graph_representation, train_eval_loader
    )
    eval_objects_loss = test_eval_objects_loss + train_eval_objects_loss
    test_eval_moser_loss = initiate_eval_moser_loss(
        "test", N_STEPS_MOSER, N_RUNS_MOSER, graph_representation, test_data, sat_data
    )
    train_eval_moser_loss = initiate_eval_moser_loss(
        "train",
        N_STEPS_MOSER,
        N_RUNS_MOSER,
        graph_representation,
        train_eval_data,
        sat_data,
    )
    eval_moser_loss = test_eval_moser_loss + train_eval_moser_loss
    eval_objects_loss = update_eval_objects_loss(params, total_loss, eval_objects_loss)
    eval_moser_loss = update_eval_moser_loss(network, params, eval_moser_loss)
    for epoch in range(NUM_EPOCHS):
        print("epoch " + str(epoch + 1) + " of " + str(NUM_EPOCHS))
        start_time = time.time()
        for counter, batch in enumerate(train_loader):
            print("batch_number", counter)
            params, opt_state = update(params, batch, opt_state)
        if model_path:
            jnp.save(
                model_path,
                np.asarray(
                    [params, [graph_representation, network_type]], dtype=object
                ),
            )
        print("model successfully saved")
        epoch_time = time.time() - start_time
        eval_objects_loss = update_eval_objects_loss(
            params, total_loss, eval_objects_loss
        )
        eval_moser_loss = update_eval_moser_loss(network, params, eval_moser_loss)

        loss_str = "Epoch {} in {:0.2f} sec".format(epoch + 1, epoch_time) + ";  "
        for eval_result in eval_objects_loss:
            loss_str = (
                loss_str
                + f"{eval_result.name}: {np.round(eval_result.results[-1],4)}"
                + "; "
            )
            if experiment_tracking == True:
                mlflow.log_metric(eval_result.name, eval_result.results[-1], step=epoch)
        for eval_result in eval_moser_loss:
            loss_str = (
                loss_str
                + f"{eval_result.name}: {np.round(eval_result.results[-1],4)}"
                + "; "
            )
            if experiment_tracking == True:
                mlflow.log_metric(eval_result.name, eval_result.results[-1], step=epoch)
        print(loss_str)
        if epoch == 0:
            t2 = time.time()
            print("took", t2 - t1, "seconds")
    if model_path:
        jnp.save(
            model_path,
            np.asarray([params, [graph_representation, network_type]], dtype=object),
        )
        print("model successfully saved")

    if img_path:
        plot_accuracy_fig(*eval_objects_loss)
        plot_accuracy_fig(*eval_moser_loss)
        plt.xlabel("epoch")
        plt.ylabel("accuracy of model / loss")
        plt.yscale("log")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        if img_path == "show":
            plt.show()
        else:
            plt.savefig(img_path + "accuracy.jpg", dpi=300, format="jpg")
    return {
        "params": params,
    }


def experiment_tracking_train(
    MODEL_REGISTRY: str,
    EXPERIMENT_NAME: str,
    batch_size: int,
    inv_temp: float,
    alpha: float,
    beta: float,
    gamma: float,
    NUM_EPOCHS: int,
    N_STEPS_MOSER: int,
    N_RUNS_MOSER: int,
    data_path: str,
    graph_representation: str,
    mlp_layers: list[int],
    network_type: str = "interaction",
    return_candidates=False,
):
    match graph_representation:
        case "LCG":
            rep = LCG
        case "VCG":
            rep = VCG

    network_definition = get_network_definition(
        network_type=network_type, graph_representation=rep
    )

    MODEL_REGISTRY = Path(MODEL_REGISTRY)
    MODEL_REGISTRY.mkdir(exist_ok=True)  # create experiments dir

    timestr = time.strftime("%Y%m%d-%H%M%S")
    params_save = Path("experiments", "params_save")
    params_save.mkdir(exist_ok=True)
    model_path = os.path.join(params_save, EXPERIMENT_NAME + timestr)
    img_path = model_path + "_plot"

    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # log key hyperparameters
        mlflow.log_params(
            {
                "inv_temp": inv_temp,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "batch_size": batch_size,
                "NUM_EPOCHS": NUM_EPOCHS,
                "N_STEPS_MOSER": N_STEPS_MOSER,
                "N_RUNS_MOSER": N_RUNS_MOSER,
                "network_definition": network_definition.__name__,
                "path_dataset": data_path,
                "graph_representation": graph_representation,
                "network_type": network_type,
                "return_candidates": return_candidates,
            }
        )

        # train and evaluate
        artifacts = train(
            batch_size,
            inv_temp,
            alpha,
            beta,
            gamma,
            NUM_EPOCHS,
            N_STEPS_MOSER,
            N_RUNS_MOSER,
            data_path,
            mlp_layers=mlp_layers,
            img_path=img_path,
            model_path=model_path,
            experiment_tracking=True,
            graph_representation=rep,
            network_type=network_type,
            return_candidates=return_candidates,
        )
        # log params which are a result of learning
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["params"], Path(dp, "params.pkl"))
            mlflow.log_artifact(dp)


if __name__ == "__main__":
    CLI(experiment_tracking_train)
