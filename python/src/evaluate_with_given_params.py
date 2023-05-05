import sys

sys.path.append("../../")

import numpy as np
from data_utils import SATTrainingDataset_LCG, SATTrainingDataset_VCG, JraphDataLoader
from model import (
    get_network_definition,
    get_model_probabilities,
)
import haiku as hk
import moser_rust
import glob
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# data_path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/GIT_SAT_ML/data/LLL_sample_one"
# data_path = "../Data/LLL_sample_one"
data_path = "../Data/blocksworld"
# data_path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/generateSAT/LLL_subset"
# data_path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/generateSAT/samples_n300"
# model_path = "../params_save/LCG_samples_large_n20230426-143758.npy"
# model_paths = ["../params_save/LCG_blocksworld20230427-170155.npy"]
# model_paths = [
#    "/Users/p403830/Downloads/params_save/LCG_blocksworld20230428-193154.npy",
#  "/Users/p403830/Downloads/params_save/LCG_blocksworldDM20230428-193154.npy",
#    "/Users/p403830/Downloads/params_save/LCG_blocksworldLLL20230428-193154.npy",
# ]
# model_paths = [
#    "../params_save/trash20230502-131140.npy",
#    "../params_save/trash20230502-131840.npy",
#    "../params_save/trash20230502-132336.npy",
# ]
model_paths = [
    "/Users/p403830/Downloads/params_save/LCG_blocksworldDM20230502-120802.npy",
    "/Users/p403830/Downloads/params_save/LCG_blocksworldLLL20230502-120802.npy",
    "/Users/p403830/Downloads/params_save/LCG_blocksworldLLL_DM20230502-120802.npy",
]

model_names = ["DM", "LLL", "DM + LLL"]
colors = ["orange", "teal", "cyan"]
# model_names = ["test"]

N_STEPS_MOSER_list = [1, 10, 100, 1000, 10000]
N_RUNS_MOSER = 5
SEED = 0


def load_model_and_test_moser(
    data_path, model_path, model_names, N_STEPS_MOSER_list, N_RUNS_MOSER, colors
):
    for j in range(len(model_paths)):
        params, model_details = np.load(model_path[j], allow_pickle=True)
        graph_representation, network_type = model_details

        if graph_representation == "LCG":
            sat_data = SATTrainingDataset_LCG(data_path)
        if graph_representation == "VCG":
            sat_data = SATTrainingDataset_VCG(data_path)

        # data_loader = JraphDataLoader(sat_data, batch_size=1, shuffle=True)

        network_definition = get_network_definition(
            network_type=network_type, graph_representation=graph_representation
        )
        network = hk.without_apply_rng(hk.transform(network_definition))

        energies_array = []  # np.zeros(len(N_STEPS_MOSER_list))
        if j == 0:
            energies_array_uniform = []  # np.zeros(len(N_STEPS_MOSER_list))
            energies_array_schoening = []  # np.zeros(len(N_STEPS_MOSER_list))
        for idx in range(len(sat_data)):
            problem_path = sat_data.instances[idx].name + ".cnf"
            problem = sat_data.get_unpadded_problem(idx)
            model_probabilities = get_model_probabilities(
                network, params, problem, graph_representation
            )
            model_probabilities = model_probabilities.ravel()
            if j == 0:
                uniform_probabilities = np.ones(len(model_probabilities)) / 2
            # print(np.max(model_probabilities), np.min(model_probabilities))
            single_energy = np.zeros(len(N_STEPS_MOSER_list))
            if j == 0:
                single_energy_uniform = np.zeros(len(N_STEPS_MOSER_list))
                single_energy_schoening = np.zeros(len(N_STEPS_MOSER_list))
            for i, N_STEPS_MOSER in enumerate(N_STEPS_MOSER_list):
                # _, _, final_energies = moser_rust.run_moser_python(
                #    problem_path, model_probabilities, N_STEPS_MOSER, N_RUNS_MOSER, SEED
                # )
                _, _, final_energies, numtry, numstep = moser_rust.run_sls_python(
                    "moser",
                    problem_path,
                    model_probabilities,
                    N_STEPS_MOSER - 1,
                    N_RUNS_MOSER,
                    SEED,
                )
                single_energy[i] = final_energies
                if j == 0:
                    (
                        _,
                        _,
                        final_energies_uniform,
                        numtry,
                        numstep,
                    ) = moser_rust.run_sls_python(
                        "moser",
                        problem_path,
                        uniform_probabilities,
                        N_STEPS_MOSER - 1,
                        N_RUNS_MOSER,
                        SEED,
                    )
                    single_energy_uniform[i] = final_energies_uniform
                    (
                        _,
                        _,
                        final_energies_schoening,
                        numtry,
                        numstep,
                    ) = moser_rust.run_sls_python(
                        "schoening",
                        problem_path,
                        uniform_probabilities,
                        N_STEPS_MOSER - 1,
                        N_RUNS_MOSER,
                        SEED,
                    )
                    single_energy_schoening[i] = final_energies_schoening
                # print("N_STEPS_MOSER", N_STEPS_MOSER, final_energies)
            # print(single_energy)
            energies_array.append(
                single_energy / problem.params[1]
            )  # += # single_energy / problem.params[1]
            if j == 0:
                energies_array_uniform.append(
                    single_energy_uniform / problem.params[1]
                )  # += single_energy_uniform / problem.params[1]
                energies_array_schoening.append(
                    single_energy_schoening / problem.params[1]
                )  # += single_energy_schoening / problem.params[1]
        energies_array_mean = np.mean(np.array(energies_array), axis=0)
        print(energies_array_mean.shape)
        # energies_array = energies_array / len(sat_data)
        # energies_array_uniform = energies_array_uniform / len(sat_data)
        # energies_array_schoening = energies_array_uniform / len(sat_data)
        # plt.plot(
        #    N_STEPS_MOSER_list,
        #    energies_array,
        #    "--o",
        #    label="model of " + model_names[j],
        # )
        plt.plot(
            N_STEPS_MOSER_list,
            np.mean(np.array(energies_array), axis=0),
            "--x",
            label="model of " + model_names[j],
            color=colors[j],
        )
        # plt.plot(
        #    N_STEPS_MOSER_list,
        #    np.max(np.array(energies_array), axis = 0),
        #    "--o",
        #    color = colors[j],
        #    alpha = 0.3
        # )
        # plt.plot(
        #    N_STEPS_MOSER_list,
        #    np.min(np.array(energies_array), axis = 0),
        #    "--o",
        #    color = colors[j],
        #    alpha = 0.3
        # )
        if j == 0:
            plt.plot(
                N_STEPS_MOSER_list,
                np.mean(np.array(energies_array_uniform), axis=0),
                "--x",
                label="uniform Moser",
                color="red",
            )
            # plt.plot(N_STEPS_MOSER_list, np.max(np.array(energies_array_uniform), axis = 0), "--o", label="uniform", color = "red", alpha = 0.3)
            # plt.plot(N_STEPS_MOSER_list, np.min(np.array(energies_array_uniform), axis = 0), "--o", label="uniform", color = "red", alpha = 0.3)
            plt.plot(
                N_STEPS_MOSER_list,
                np.mean(np.array(energies_array_schoening), axis=0),
                "--x",
                label="Schoening",
                color="blue",
            )
            # plt.plot(N_STEPS_MOSER_list, np.max(np.array(energies_array_schoening), axis = 0), "--o", label="uniform", color = "blue", alpha = 0.3)
            # plt.plot(N_STEPS_MOSER_list, np.min(np.array(energies_array_schoening), axis = 0), "--o", label="uniform", color = "blue", alpha = 0.3)
    plt.xlabel("N_STEPS_MOSER")
    plt.ylabel("# violated clauses / m")
    plt.xscale("log")
    plt.legend()
    plt.grid()
    plt.show()


def load_model_and_test_moser2(
    data_path, model_paths_list, model_names, N_STEPS_MOSER, N_RUNS_MOSER
):
    total_steps = []
    for j in range(len(model_paths)):
        if model_paths_list[j] != "uniform":
            params, model_details = np.load(model_paths_list[j], allow_pickle=True)
            graph_representation, network_type = model_details

            if graph_representation == "LCG":
                sat_data = SATTrainingDataset_LCG(data_path)
            if graph_representation == "VCG":
                sat_data = SATTrainingDataset_VCG(data_path)

            # data_loader = JraphDataLoader(sat_data, batch_size=1, shuffle=True)

            network_definition = get_network_definition(
                network_type=network_type, graph_representation=graph_representation
            )
            network = hk.without_apply_rng(hk.transform(network_definition))

        model_steps = np.zeros(len(sat_data))

        for idx in range(len(sat_data)):
            problem_path = sat_data.instances[idx].name + ".cnf"
            problem = sat_data.get_unpadded_problem(idx)
            if model_paths_list[j] != "uniform":
                model_probabilities = get_model_probabilities(
                    network, params, problem, graph_representation
                )
                model_probabilities = model_probabilities.ravel()
            else:
                model_probabilities = np.ones(len(model_probabilities)) / 2
            _, _, final_energies, numtry, numstep = moser_rust.run_sls_python(
                "moser",
                problem_path,
                model_probabilities,
                N_STEPS_MOSER - 1,
                N_RUNS_MOSER,
                SEED,
            )
            model_steps[idx] = numstep
        total_steps.append(model_steps)
    print(total_steps[0])
    print(total_steps[1])
    plt.scatter(total_steps[0], total_steps[1], alpha=0.4)
    x = np.arange(
        0.9 * np.min(np.ravel(total_steps)), 1.1 * np.max(np.ravel(total_steps)), 1000
    )
    plt.plot(x, x, "--", label="f(x)=x", color="gray")
    plt.ylabel("# steps model " + model_names[0])
    plt.xlabel("# steps model " + model_names[1])
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()
    # idea: use degree of improvement (see Susa, Nishimori paper!)


load_model_and_test_moser(
    data_path, model_paths, model_names, N_STEPS_MOSER_list, N_RUNS_MOSER, colors
)


data_path = "../Data/blocksworld"
model_paths_list = [
    "uniform",
    "/Users/p403830/Downloads/params_save/LCG_blocksworldLLL_DM20230502-120802.npy",
]
load_model_and_test_moser2(data_path, model_paths_list, model_names, 1000000, 5)
