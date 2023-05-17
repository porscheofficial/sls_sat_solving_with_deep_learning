from functools import partial
import sys
import mlflow
from pathlib import Path
import tempfile
import joblib
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import time
from torch.utils import data
from torch import Generator
import matplotlib.pyplot as plt

sys.path.append("../../")

from python.src.data_utils import SATTrainingDataset, JraphDataLoader
from python.src.sat_representations import VCG, LCG, SATRepresentation
from python.src.model import get_network_definition
from python.src.train_utils import (
    evaluate_moser_rust,
    EvalResults,
    plot_accuracy_fig,
    initiate_eval_objects_loss,
    update_eval_objects_loss,
    initiate_eval_moser_loss,
    update_eval_moser_loss,
)


def train(
    batch_size,
    f,
    alpha,
    beta,
    gamma,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    N_RUNS_MOSER,
    path,
    img_path=False,
    model_path=False,
    experiment_tracking=False,
    graph_representation=SATRepresentation,
    network_type="interaction",
    return_candidates=False,
):
    sat_data = SATTrainingDataset(
        path, graph_representation, return_candidates=return_candidates
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
    network = hk.without_apply_rng(hk.transform(network_definition))
    params = network.init(jax.random.PRNGKey(42), sat_data[0][0].graph)

    opt_init, opt_update = optax.adam(1e-3)
    opt_state = opt_init(params)

    @partial(jax.jit, static_argnums=(2, 3, 4, 5, 6))
    def total_loss(
        params,
        batch,
        f: float,
        alpha: float,
        beta: float,
        gamma: float,
        rep: SATRepresentation,
    ):
        (mask, graph, neighbors_list), (candidates, energies) = batch
        decoded_nodes = network.apply(params, graph)
        prediction_loss = (
            alpha * rep.prediction_loss(decoded_nodes, mask, candidates, energies, f)
            if alpha > 0
            else 0.0
        )
        local_lovasz_loss = (
            beta * rep.local_lovasz_loss(decoded_nodes, mask, graph, neighbors_list)
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
            params, batch, f, alpha, beta, gamma, graph_representation
        )
        updates, opt_state = opt_update(g, opt_state)
        return optax.apply_updates(params, updates), opt_state

    print("Entering training loop")

    test_eval_objects_loss = initiate_eval_objects_loss(
        "test", f, alpha, beta, gamma, graph_representation, test_loader
    )
    train_eval_objects_loss = initiate_eval_objects_loss(
        "train", f, alpha, beta, gamma, graph_representation, train_eval_loader
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
    print(eval_moser_loss)
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

        # test_LLL_loss = evaluate(test_loader, local_lovasz_loss, graph_representation)
        # train_LLL_loss = evaluate(
        #     train_eval_loader, local_lovasz_loss, graph_representation
        # )
        # test_pred_loss = evaluate(test_loader, prediction_loss, graph_representation)
        # train_pred_loss = evaluate(
        #     train_eval_loader, prediction_loss, graph_representation
        # )
        # test_entropy_loss = evaluate(test_loader, entropy_loss, graph_representation)
        # train_entropy_loss = evaluate(
        #     train_eval_loader, entropy_loss, graph_representation
        # )

        # test_eval.results.append(test_LLL_loss + test_pred_loss + test_entropy_loss)
        # train_eval.results.append(train_LLL_loss + train_pred_loss + train_entropy_loss)
        # test_eval_lll.results.append(test_LLL_loss)
        # train_eval_lll.results.append(train_LLL_loss)
        # test_eval_dm.results.append(test_pred_loss)
        # train_eval_dm.results.append(train_pred_loss)

        # test_moser_energies, _ = evaluate_moser_rust(
        #     test_data,
        # )
        # train_moser_energies, _ = evaluate_moser_rust(
        #     train_eval_data,
        # )

        # test_moser_eval.results.append(test_moser_energies)
        # train_moser_eval.results.append(train_moser_energies)
        # test_entropy_eval.results.append(test_entropy_loss)
        # train_entropy_eval.results.append(train_entropy_loss)
        # test_moser_baseline.results.append(moser_baseline_test)
        # train_moser_baseline.results.append(moser_baseline_train)

        # loss_str = "Epoch {} in {:0.2f} sec".format(epoch, epoch_time) + ";  "
        # for eval_result in eval_objects:
        #     loss_str = (
        #         loss_str
        #         + f"{eval_result.name}: {np.round(eval_result.results[-1],4)}"
        #         + "; "
        #     )
        #     if experiment_tracking == True:
        #         mlflow.log_metric(eval_result.name, eval_result.results[-1], step=epoch)
        # print(loss_str)
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
        plot_accuracy_fig(*test_eval_objects_loss)
        plot_accuracy_fig(*train_eval_objects_loss)
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
    MODEL_REGISTRY,
    EXPERIMENT_NAME,
    batch_size,
    f,
    alpha,
    beta,
    gamma,
    NUM_EPOCHS,
    N_STEPS_MOSER,
    N_RUNS_MOSER,
    path,
    img_path=False,
    model_path=False,
    graph_representation=SATRepresentation,
    network_type="interaction",
):
    network_definition = get_network_definition(
        network_type=network_type, graph_representation=graph_representation
    )
    Path(MODEL_REGISTRY).mkdir(exist_ok=True)  # create experiments dir
    mlflow.set_tracking_uri("file://" + str(MODEL_REGISTRY.absolute()))
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run():
        # log key hyperparameters
        mlflow.log_params(
            {
                "f": f,
                "alpha": alpha,
                "beta": beta,
                "gamma": gamma,
                "batch_size": batch_size,
                "NUM_EPOCHS": NUM_EPOCHS,
                "N_STEPS_MOSER": N_STEPS_MOSER,
                "N_RUNS_MOSER": N_RUNS_MOSER,
                "network_definition": network_definition.__name__,
                "path_dataset": path,
                "graph_representation": graph_representation,
                "network_type": network_type,
            }
        )

        # train and evaluate
        artifacts = train(
            batch_size,
            f,
            alpha,
            beta,
            gamma,
            NUM_EPOCHS,
            N_STEPS_MOSER,
            N_RUNS_MOSER,
            path,
            img_path=img_path,
            model_path=model_path,
            experiment_tracking=True,
            graph_representation=graph_representation,
            network_type=network_type,
        )
        # log params which are a result of learning
        with tempfile.TemporaryDirectory() as dp:
            joblib.dump(artifacts["params"], Path(dp, "params.pkl"))
            mlflow.log_artifact(dp)


if __name__ == "__main__":
    NUM_EPOCHS = 20  # 10
    f = 0.0000001
    alpha = 1
    beta = 0
    gamma = 0
    batch_size = 1
    # path = "../Data/mini"
    # path = "../Data/LLL_sample_one_combination"
    # path = "../Data/LLL_sample_one"
    # path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/generateSAT/LLL_subset"
    # path = "../Data/blocksworld"
    # path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/GIT_SAT_ML/data/BroadcastTestSet2"
    path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/GIT_SAT_ML/data/BroadcastTestSet"
    # path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/generateSAT/samples_LLL_n80/"
    N_STEPS_MOSER = 100
    N_RUNS_MOSER = 5
    SEED = 0
    graph_representation = "VCG"
    network_type = "interaction"
    # network_definition = get_network_definition(network_type = network_type, graph_representation = graph_representation) #network_definition_interaction_new

    MODEL_REGISTRY = Path("../../mlrun_save")
    EXPERIMENT_NAME = "VCG_broadcast"

    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_path = "../params_save/" + EXPERIMENT_NAME + timestr
    img_path = model_path + "_plot"

    match graph_representation:
        case "LCG":
            rep = LCG
        case "VCG":
            rep = VCG

    experiment_tracking_train(
        MODEL_REGISTRY,
        EXPERIMENT_NAME,
        batch_size,
        f,
        alpha,
        beta,
        gamma,
        NUM_EPOCHS,
        N_STEPS_MOSER,
        N_RUNS_MOSER,
        path,
        img_path=img_path,
        model_path=model_path,
        rep=rep,
        network_type=network_type,
    )
