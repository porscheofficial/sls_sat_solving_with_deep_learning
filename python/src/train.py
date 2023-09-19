"""Train the graph neural network using the method in this file."""
from functools import partial
import sys
import time
from pathlib import Path
import tempfile
import os
from typing import Any
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


from python.src.data_utils import SATTrainingDataset, JraphDataLoader
from python.src.sat_representations import VCG, LCG
from python.src.model import get_network_definition
from python.src.train_utils import (
    plot_accuracy_fig,
    update_eval_objects_loss,
    update_eval_moser_loss,
    initiate_eval_objetcts_train_test,
    initiate_eval_moser_train_test,
)

sys.path.append("../../")


def train(
    batch_size: int,
    inv_temp: float,
    alpha: float,
    beta: float,
    gamma: float,
    num_epochs: int,
    n_steps_moser: int,
    n_runs_moser: int,
    data_path,
    graph_representation: str,
    network_type,
    mlp_layers,
    img_path=False,
    model_path=False,
    experiment_tracking=False,
    return_candidates=False,
    initial_learning_rate=0.001,
    final_learning_rate=0.001,
):
    """Execute the training loop.

    Args:
        batch_size (int): batch size that is used
        inv_temp (float): inverse temperature used in Gibbs Loss
        alpha (float): prefactor used for Gibbs loss
        beta (float): prefactor used for LLL loss
        gamma (float): prefactor used for alternative LLL loss
        num_epochs (int): _description_
        n_steps_moser (int): number of steps used in MT algorithm statistics
        n_runs_moser (int): number of runs used in MT algorithm statistics
        path (str): path pointing to training dataset (this is split into train and test set)
        graph_representation (SATRepresentation): SATRepresentation used. Either LCG or VCG.
        network_type (str): either "interaction" or "GCN".
        mlp_layers (array): size of mlp_layers. For example: [200,200]
        img_path (bool, optional): path where the plot is saved. Defaults to False.
        model_path (bool, optional): path where the model is saved. Defaults to False.
        experiment_tracking (bool, optional): decide whether experiment tracking is done using MLflow. Defaults to False.
        return_candidates (bool, optional): decide whether candidates are used for Gibbs loss or only the solution. Defaults to False.
        initial_learning_rate (float, optional): initial learning rate that is chosen. Defaults to 0.001.
        final_learning_rate (float, optional):  final learning rate that is chosen. Note that the learning rate decays from the initial learing rate exponentially to the final learning rate over the epochs. Defaults to 0.001.

    Returns:
        @TODO: type: final params of the net
    """
    match graph_representation:
        case "LCG":
            graph_representation_rep: Any = LCG
        case "VCG":
            graph_representation_rep = VCG

    include_constraint_graph = (
        beta + gamma > 0
    )  # we calculate the constraint graphs only if we use it to calculate the lll loss

    sat_data = SATTrainingDataset(
        data_path,
        graph_representation_rep,
        return_candidates=return_candidates,
        include_constraint_graph=include_constraint_graph,
    )
    train_data, test_data = data.random_split(
        sat_data, [0.8, 0.2], generator=Generator().manual_seed(0)
    )
    train_eval_data, _ = data.random_split(
        train_data, [0.2, 0.8], generator=Generator().manual_seed(0)
    )
    train_loader = JraphDataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = JraphDataLoader(test_data, batch_size=batch_size)
    train_eval_loader = JraphDataLoader(train_eval_data, batch_size=batch_size)

    network_definition = get_network_definition(
        network_type=network_type, graph_representation=graph_representation_rep
    )
    network_definition = partial(network_definition, mlp_layers=mlp_layers)
    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)  # type: ignore[attr-defined]

    # use a schedule function for the ADAM optimizer
    tot_steps = int(num_epochs * np.ceil(len(train_data) / batch_size))
    decay_rate = final_learning_rate / initial_learning_rate
    exponential_decay_scheduler = optax.exponential_decay(
        init_value=initial_learning_rate,
        transition_steps=tot_steps,
        decay_rate=decay_rate,
        transition_begin=int(tot_steps * 0.05),
        staircase=False,
    )

    opt_init, opt_update = optax.adam(learning_rate=exponential_decay_scheduler)
    opt_state = opt_init(params)

    # opt_init, opt_update = optax.adam(1e-6)
    # opt_state = opt_init(params)

    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
    def total_loss(
        params,
        batch,
        inv_temp: float,
        alpha: float,
        beta: float,
        gamma: float,
        rep,
    ):
        (mask, graph, constraint_graph, constraint_mask), (candidates, energies) = batch
        decoded_nodes = network.apply(params, graph)  # type: ignore[attr-defined]
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
        alt_local_lovasz_loss = (
            gamma
            * rep.alt_local_lovasz_loss(
                decoded_nodes, mask, graph, constraint_graph, constraint_mask
            )
            if gamma > 0
            else 0.0
        )

        return (
            prediction_loss + local_lovasz_loss + alt_local_lovasz_loss
        )  # + entropy_loss

    @jax.jit
    def update(params, batch, opt_state):
        gradient = jax.grad(total_loss)(
            params, batch, inv_temp, alpha, beta, gamma, graph_representation_rep
        )
        updates, opt_state = opt_update(gradient, opt_state)
        return optax.apply_updates(params, updates), opt_state

    print("Entering training loop")

    eval_objects_loss = initiate_eval_objetcts_train_test(
        inv_temp,
        alpha,
        beta,
        gamma,
        graph_representation_rep,
        test_loader,
        train_eval_loader,
    )
    eval_moser_loss = initiate_eval_moser_train_test(
        n_steps_moser,
        n_runs_moser,
        graph_representation_rep,
        test_data,
        train_eval_data,
        sat_data,
    )

    eval_objects_loss = update_eval_objects_loss(params, total_loss, eval_objects_loss)
    if n_steps_moser != 0:
        eval_moser_loss = update_eval_moser_loss(network, params, eval_moser_loss)
    for epoch in range(num_epochs):
        print("epoch " + str(epoch + 1) + " of " + str(num_epochs))
        start_time = time.time()
        for counter, batch in enumerate(train_loader):
            print("batch_number", counter)
            params, opt_state = update(params, batch, opt_state)
        if model_path:
            jnp.save(
                model_path,
                np.asarray(
                    [
                        params,
                        [
                            inv_temp,
                            alpha,
                            beta,
                            gamma,
                            mlp_layers,
                            graph_representation_rep,
                            network_type,
                            return_candidates,
                        ],
                    ],
                    dtype=object,
                ),
            )
        print("model successfully saved")
        epoch_time = time.time() - start_time
        eval_objects_loss = update_eval_objects_loss(
            params, total_loss, eval_objects_loss
        )
        if n_steps_moser != 0:
            eval_moser_loss = update_eval_moser_loss(network, params, eval_moser_loss)

        loss_str = f"Epoch {epoch} in {np.round(epoch_time, 2)} sec;  "
        for eval_result in eval_objects_loss:
            loss_str = (
                loss_str
                + f"{eval_result.name}: {np.round(eval_result.results[-1],6)}; "
            )
            if experiment_tracking:
                mlflow.log_metric(eval_result.name, eval_result.results[-1], step=epoch)
        if n_steps_moser != 0:
            for eval_result in eval_moser_loss:
                loss_str = (
                    loss_str
                    + f"{eval_result.name}: {np.round(eval_result.results[-1],4)}; "
                    + "; "
                )
                if experiment_tracking:
                    mlflow.log_metric(
                        eval_result.name, eval_result.results[-1], step=epoch
                    )
        print(loss_str)
    if model_path:
        jnp.save(
            model_path,
            np.asarray(
                [
                    params,
                    [
                        inv_temp,
                        alpha,
                        beta,
                        gamma,
                        mlp_layers,
                        graph_representation_rep,
                        network_type,
                        return_candidates,
                    ],
                ],
                dtype=object,
            ),
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
    model_registry: str,
    experiment_name: str,
    batch_size: int,
    inv_temp: float,
    alpha: float,
    beta: float,
    gamma: float,
    num_epochs: int,
    n_steps_moser: int,
    n_runs_moser: int,
    data_path: str,
    graph_representation: str,
    mlp_layers: list[int],
    network_type: str = "interaction",
    return_candidates=True,
    initial_learning_rate=0.001,
    final_learning_rate=0.001,
):
    """Training loop that is tracked by MLflow.

    Args:
        MODEL_REGISTRY (str): path where experiment tracking is saved
        EXPERIMENT_NAME (str): name of the experiment in MLflow

        batch_size (int): batch size that is used
        inv_temp (float): inverse temperature used in Gibbs Loss
        alpha (float): prefactor used for Gibbs loss
        beta (float): prefactor used for LLL loss
        gamma (float): prefactor used for alternative LLL loss
        num_epochs (int): _description_
        n_steps_moser (int): number of steps used in MT algorithm statistics
        n_runs_moser (int): number of runs used in MT algorithm statistics
        data_path (str): path pointing to training dataset (this is split into train and test set)
        graph_representation (str): Representation used. Either "LCG" or "VCG"
        mlp_layers (array): size of mlp_layers. For example: [200,200]
        network_type (str): either "interaction" for interaction network or "GCN" for Graph convolutional network (not tested!)
        img_path (bool, optional): path where the plot is saved that contains the loss function plot as a function of the epochs. Defaults to False.
        return_candidates (bool, optional): decide whether candidates are used for Gibbs loss or only the solution. Defaults to False.
        initial_learning_rate (float, optional): initial learning rate that is chosen. Defaults to 0.001.
        final_learning_rate (float, optional): final learning rate that is chosen. Note that the learning rate decays from the initial learing rate exponentially to the final learning rate over the epochs. Defaults to 0.001.

    Raises:
            ValueError: if no proper graph representation is chosen, raise a value error
    """
    # match graph_representation:
    #    case "LCG":
    #        graph_representation_rep = LCG
    #   case "VCG":
    #        graph_representation_rep = VCG

    if graph_representation == "LCG":
        graph_representation_rep: Any = LCG
    elif graph_representation == "VCG":
        graph_representation_rep = VCG

    network_definition = get_network_definition(
        network_type=network_type, graph_representation=graph_representation_rep
    )

    model_registry_path = Path(model_registry)
    model_registry_path.mkdir(exist_ok=True)  # create experiments dir

    timestr = time.strftime("%Y%m%d-%H%M%S")
    params_save = Path("experiments", "params_save")
    params_save.mkdir(exist_ok=True)
    model_path = os.path.join(params_save, experiment_name + timestr)
    img_path = model_path + "_plot"

    mlflow.set_tracking_uri("file://" + str(model_registry_path.absolute()))
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run():
        # log key hyperparameters
        mlflow.log_params(
            {
                "inv_temp": inv_temp,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "batch_size": batch_size,
                "NUM_EPOCHS": num_epochs,
                "N_STEPS_MOSER": n_steps_moser,
                "N_RUNS_MOSER": n_runs_moser,
                "network_definition": network_definition.__name__,
                "path_dataset": data_path,
                "graph_representation": graph_representation_rep,
                "network_type": network_type,
                "return_candidates": return_candidates,
                "mlp_layers": mlp_layers,
                "initial_learning_rate": initial_learning_rate,
                "final_learning_rate": final_learning_rate,
            }
        )

        # train and evaluate
        artifacts = train(
            batch_size,
            inv_temp,
            alpha,
            beta,
            gamma,
            num_epochs,
            n_steps_moser,
            n_runs_moser,
            data_path,
            mlp_layers=mlp_layers,
            img_path=img_path,
            model_path=model_path,
            experiment_tracking=True,
            graph_representation=graph_representation,
            network_type=network_type,
            return_candidates=return_candidates,
            initial_learning_rate=initial_learning_rate,
            final_learning_rate=final_learning_rate,
        )
        # log params which are a result of learning
        with tempfile.TemporaryDirectory() as dump:
            joblib.dump(artifacts["params"], Path(dump, "params.pkl"))
            mlflow.log_artifact(dump)


if __name__ == "__main__":
    CLI(experiment_tracking_train)
