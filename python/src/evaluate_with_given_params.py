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
import time

# data_path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/GIT_SAT_ML/data/LLL_sample_one"
# data_path = "../Data/LLL_sample_one"
# data_path = "../Data/blocksworld"

data_path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/GIT_SAT_ML/data/BroadcastTestSet_subset"
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
"""
model_paths = [
    "/Users/p403830/Downloads/params_save/LCG_blocksworldDM20230502-120802.npy",
    "/Users/p403830/Downloads/params_save/LCG_blocksworldLLL20230502-120802.npy",
    "/Users/p403830/Downloads/params_save/LCG_blocksworldLLL_DM20230502-120802.npy",
    "/Users/p403830/Downloads/params_save/VCG_blocksworldDM20230504-150901.npy",
    "/Users/p403830/Downloads/params_save/VCG_blocksworldLLL20230504-150901.npy",
    "/Users/p403830/Downloads/params_save/VCG_blocksworldLLL_DM20230504-150901.npy"
]

model_names = ["LCG - DM", "LCG - LLL", "LCG - DM + LLL", "VCG - DM", "VCG - LLL", "VCG - DM + LLL"]
colors = ["orange", "teal", "cyan", "magenta", "gray", "lightgreen"]
model_paths = [
    "/Users/p403830/Downloads/params_save/VCG_blocksworldLLL_DM20230504-150901.npy"
]
"""
# model_paths = [
#    "/Users/p403830/Downloads/params_save/VCG_n300LLL_DM20230505-140002.npy",
#    "/Users/p403830/Downloads/params_save/VCG_n300DM20230505-140002.npy",
#    "/Users/p403830/Downloads/params_save/VCG_n300LLL20230505-140002.npy"
# ]
# model_names = ["VCG - DM + LLL", "VCG - DM", "VCG - LLL"]
# colors = ["orange", "teal", "cyan"]

# model_names = ["test"]

# N_STEPS_MOSER_list = [1, 10, 100, 1000]
# N_RUNS_MOSER = 5
SEED = 0


def load_model_and_test_moser(
    data_path, model_paths, model_names, N_STEPS_MOSER_list, N_RUNS_MOSER, colors
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
                _, _, final_energies, numtry, numstep, traj = moser_rust.run_sls_python(
                    "moser",
                    problem_path,
                    model_probabilities,
                    N_STEPS_MOSER - 1,
                    N_RUNS_MOSER,
                    SEED,
                    True,
                )
                single_energy[i] = final_energies
                if j == 0:
                    (
                        _,
                        _,
                        final_energies_uniform,
                        numtry,
                        numstep,
                        traj,
                    ) = moser_rust.run_sls_python(
                        "moser",
                        problem_path,
                        uniform_probabilities,
                        N_STEPS_MOSER - 1,
                        N_RUNS_MOSER,
                        SEED,
                        True,
                    )
                    single_energy_uniform[i] = final_energies_uniform
                    (
                        _,
                        _,
                        final_energies_schoening,
                        numtry,
                        numstep,
                        traj,
                    ) = moser_rust.run_sls_python(
                        "schoening",
                        problem_path,
                        uniform_probabilities,
                        N_STEPS_MOSER - 1,
                        N_RUNS_MOSER,
                        SEED,
                        True,
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


def get_padded_trajs(traj, N_STEPS):
    array_traj = []
    for i in range(len(traj)):
        array_traj.append(np.pad(traj[i], (0, N_STEPS - len(traj[i]))))
    return np.array(array_traj)


def load_model_and_test_moser_traj(
    data_path,
    model_paths,
    model_names,
    N_STEPS_MOSER,
    N_RUNS_MOSER,
    colors,
    path_trajs=False,
):
    total_array = []
    for j in range(len(model_paths)):
        print(model_names[j])
        params, model_details = np.load(model_paths[j], allow_pickle=True)
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
            print("problem ", idx + 1, "of ", len(sat_data))
            problem_path = sat_data.instances[idx].name + ".cnf"
            problem = sat_data.get_unpadded_problem(idx)
            model_probabilities = get_model_probabilities(
                network, params, problem, graph_representation
            )
            model_probabilities = model_probabilities.ravel()
            if j == 0:
                uniform_probabilities = np.ones(len(model_probabilities)) / 2
            # print(np.max(model_probabilities), np.min(model_probabilities))
            single_traj = []  # np.zeros(len(N_STEPS_MOSER_list))
            if j == 0:
                single_traj_uniform = []  # np.zeros(len(N_STEPS_MOSER_list))
                single_traj_schoening = []  # np.zeros(len(N_STEPS_MOSER_list))

            _, _, final_energies, numtry, numstep, traj = moser_rust.run_sls_python(
                "moser",
                problem_path,
                model_probabilities,
                N_STEPS_MOSER - 1,
                N_RUNS_MOSER,
                SEED,
                True,
            )
            traj = get_padded_trajs(traj, N_STEPS_MOSER)
            single_traj.append(np.mean(traj, axis=0) / problem.params[1])
            if j == 0:
                (
                    _,
                    _,
                    final_energies_uniform,
                    numtry,
                    numstep,
                    traj_uniform,
                ) = moser_rust.run_sls_python(
                    "moser",
                    problem_path,
                    uniform_probabilities,
                    N_STEPS_MOSER - 1,
                    N_RUNS_MOSER,
                    SEED,
                    True,
                )
                traj_uniform = get_padded_trajs(traj_uniform, N_STEPS_MOSER)
                single_traj_uniform.append(
                    np.mean(traj_uniform, axis=0) / problem.params[1]
                )
                (
                    _,
                    _,
                    final_energies_schoening,
                    numtry,
                    numstep,
                    traj_schoening,
                ) = moser_rust.run_sls_python(
                    "schoening",
                    problem_path,
                    uniform_probabilities,
                    N_STEPS_MOSER - 1,
                    N_RUNS_MOSER,
                    SEED,
                    True,
                )
                traj_schoening = get_padded_trajs(traj_schoening, N_STEPS_MOSER)
                # print(np.asarray(traj_schoening))
                single_traj_schoening.append(
                    np.mean(np.asarray(traj_schoening), axis=0) / problem.params[1]
                )
            # print(single_energy)
            # print(single_traj)
            # print(np.array(single_traj).shape)
            # print(np.pad(np.array(single_traj)[0], (0, N_STEPS_MOSER - len(single_traj[0]))))
            # print(np.pad(np.array(single_traj)[0], (0, N_STEPS_MOSER - len(single_traj[0]))).shape)
            # print(len(single_traj[0]))
            energies_array.append(
                np.pad(
                    np.array(single_traj)[0], (0, N_STEPS_MOSER - len(single_traj[0]))
                )
            )  # += # single_energy / problem.params[1]
            if j == 0:
                # print(len(single_traj_uniform[0]))
                energies_array_uniform.append(
                    np.pad(
                        np.array(single_traj_uniform)[0],
                        (0, N_STEPS_MOSER - len(single_traj_uniform[0])),
                    )
                )  # += single_energy_uniform / problem.params[1]
                # print(len(single_traj_schoening[0]))
                energies_array_schoening.append(
                    np.pad(
                        np.array(single_traj_schoening)[0],
                        (0, N_STEPS_MOSER - len(single_traj_schoening[0])),
                    )
                )  # += single_energy_schoening / problem.params[1]
        energies_array = np.array(energies_array, dtype=object)
        # print(energies_array.shape)
        energies_array_mean = np.mean(energies_array, axis=0)
        # print(energies_array_mean.shape)
        energies_array_uniform = np.array(energies_array_uniform)
        energies_array_uniform_mean = np.mean(energies_array_uniform, axis=0)
        energies_array_schoening = np.array(energies_array_schoening)
        energies_array_schoening_mean = np.mean(energies_array_schoening, axis=0)
        # energies_array = energies_array / len(sat_data)
        # energies_array_uniform = energies_array_uniform / len(sat_data)
        # energies_array_schoening = energies_array_uniform / len(sat_data)
        # plt.plot(
        #    N_STEPS_MOSER_list,
        #    energies_array,
        #    "--o",
        #    label="model of " + model_names[j],
        # )
        if j == 0:
            total_array.append(energies_array_uniform_mean)
            plt.plot(
                np.arange(0, len(energies_array_uniform_mean), 1),
                energies_array_uniform_mean,
                "--",
                label="uniform Moser",
                color="red",
            )
            # plt.plot(N_STEPS_MOSER_list, np.max(np.array(energies_array_uniform), axis = 0), "--o", label="uniform", color = "red", alpha = 0.3)
            # plt.plot(N_STEPS_MOSER_list, np.min(np.array(energies_array_uniform), axis = 0), "--o", label="uniform", color = "red", alpha = 0.3)
            plt.plot(
                np.arange(0, len(energies_array_schoening_mean), 1),
                energies_array_schoening_mean,
                "--",
                label="Schoening",
                color="blue",
            )
            total_array.append(energies_array_schoening_mean)
            # plt.plot(N_STEPS_MOSER_list, np.max(np.array(energies_array_schoening), axis = 0), "--o", label="uniform", color = "blue", alpha = 0.3)
            # plt.plot(N_STEPS_MOSER_list, np.min(np.array(energies_array_schoening), axis = 0), "--o", label="uniform", color = "blue", alpha = 0.3)
        plt.plot(
            np.arange(0, len(energies_array_mean), 1),
            energies_array_mean,
            "-",
            label="model of " + model_names[j],
            color=colors[j],
        )
        total_array.append(energies_array_mean)
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

    plt.xlabel("N_STEPS_MOSER")
    plt.ylabel("# violated clauses / m")
    plt.xscale("log")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    if path_trajs:
        np.save(
            path_trajs,
            np.array(total_array, dtype=object),
        )
    plt.show()


def load_model_and_test_moser2(
    data_path, model_paths_list, model_names, N_STEPS_MOSER, N_RUNS_MOSER
):
    SEED = 0
    total_steps = []
    n_list = []
    alpha_list = []
    for j in range(len(model_paths_list)):
        if model_paths_list[j] != "schoening" and model_paths_list[j] != "uniform":
            params, model_details = np.load(model_paths_list[j], allow_pickle=True)
            graph_representation, network_type = model_details

            if graph_representation == "LCG":
                sat_data = SATTrainingDataset_LCG(data_path)
            if graph_representation == "VCG":
                sat_data = SATTrainingDataset_VCG(data_path)
            network_definition = get_network_definition(
                network_type=network_type, graph_representation=graph_representation
            )
            network = hk.without_apply_rng(hk.transform(network_definition))
        else:
            sat_data = SATTrainingDataset_LCG(data_path)

            # data_loader = JraphDataLoader(sat_data, batch_size=1, shuffle=True)

        model_steps = np.zeros(len(sat_data))

        for idx in range(len(sat_data)):
            print("problem ", idx + 1, "of ", len(sat_data))
            problem_path = sat_data.instances[idx].name + ".cnf"
            problem = sat_data.get_unpadded_problem(idx)
            if j == 0:
                n_list.append(problem.params[0])
                alpha_list.append(problem.params[1] / problem.params[0])
            if model_paths_list[j] == "schoening":
                model_probabilities = np.ones(problem.params[0]) / 2
                _, _, final_energies, numtry, numstep, traj = moser_rust.run_sls_python(
                    "schoening",
                    problem_path,
                    model_probabilities,
                    N_STEPS_MOSER,
                    N_RUNS_MOSER,
                    SEED,
                    True,
                )
            elif model_paths_list[j] == "uniform":
                model_probabilities = np.ones(problem.params[0]) / 2
                _, _, final_energies, numtry, numstep, traj = moser_rust.run_sls_python(
                    "moser",
                    problem_path,
                    model_probabilities,
                    N_STEPS_MOSER,
                    N_RUNS_MOSER,
                    SEED,
                    True,
                )
            else:
                model_probabilities = get_model_probabilities(
                    network, params, problem, graph_representation
                )
                model_probabilities = model_probabilities.ravel()
                _, _, final_energies, numtry, numstep, traj = moser_rust.run_sls_python(
                    "moser",
                    problem_path,
                    model_probabilities,
                    N_STEPS_MOSER,
                    N_RUNS_MOSER,
                    SEED,
                    True,
                )
            print(numstep)
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
    plt.xlim(1, np.max(np.ravel(total_steps)) * 1.2)
    plt.ylim(1, np.max(np.ravel(total_steps)) * 1.2)
    plt.show()
    return total_steps, n_list, alpha_list

    # idea: use degree of improvement (see Susa, Nishimori paper!)


def load_model_and_test_moser_single(
    data_path, model_path, N_STEPS_MOSER, N_RUNS_MOSER, path_save=False
):
    energies_array = []
    n_array = []
    alpha_array = []
    total_steps = []
    if model_path != "schoening" and model_path != "uniform":
        params, model_details = np.load(model_path, allow_pickle=True)
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
    else:
        sat_data = SATTrainingDataset_LCG(data_path)

    for idx in range(len(sat_data)):
        print("problem ", idx + 1, "of ", len(sat_data))
        problem_path = sat_data.instances[idx].name + ".cnf"
        problem = sat_data.get_unpadded_problem(idx)

        n_array.append(problem.params[0])
        alpha_array.append(problem.params[1] / problem.params[0])
        if model_path != "schoening" and model_path != "uniform":
            model_probabilities = get_model_probabilities(
                network, params, problem, graph_representation
            )
            model_probabilities = model_probabilities.ravel()
        else:
            model_probabilities = np.ones(problem.params[0]) / 2

        # print(np.max(model_probabilities), np.min(model_probabilities))
        single_traj = []  # np.zeros(len(N_STEPS_MOSER_list))
        if model_path != "schoening":
            _, _, final_energies, numtry, numstep, traj = moser_rust.run_sls_python(
                "moser",
                problem_path,
                model_probabilities,
                N_STEPS_MOSER - 1,
                N_RUNS_MOSER,
                SEED,
                True,
            )
        else:
            _, _, final_energies, numtry, numstep, traj = moser_rust.run_sls_python(
                "schoening",
                problem_path,
                model_probabilities,
                N_STEPS_MOSER - 1,
                N_RUNS_MOSER,
                SEED,
                True,
            )
        total_steps.append(numstep)
        traj = get_padded_trajs(traj, N_STEPS_MOSER)
        single_traj.append(np.mean(traj, axis=0) / problem.params[1])

        energies_array.append(
            np.pad(np.array(single_traj)[0], (0, N_STEPS_MOSER - len(single_traj[0])))
        )  # += # single_energy / problem.params[1]

    energies_array = np.array(energies_array, dtype=object)
    energies_array_mean = np.mean(energies_array, axis=0)
    total_array = [[model_path], n_array, alpha_array, energies_array_mean, total_steps]
    if path_save:
        np.save(path_save, np.array(total_array, dtype=object))
    return total_array


# load_model_and_test_moser(
#    data_path, model_paths, model_names, N_STEPS_MOSER_list, N_RUNS_MOSER, colors
# )
""""
N_STEPS_MOSER = 100000

timestamp = time.time()
path_trajs = "/Users/p403830/Desktop/" + str(timestamp) + "model_n300_test_BroadcastTestSet_subset.npy"
load_model_and_test_moser_traj(
    data_path, model_paths, model_names, N_STEPS_MOSER, N_RUNS_MOSER, colors
)

# data_path = "../Data/blocksworld"
data_path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/GIT_SAT_ML/data/BroadcastTestSet_subset"
model_paths_list = [
    "schoening",
    "/Users/p403830/Downloads/params_save/VCG_blocksworldLLL_DM20230504-150901.npy",
]
model_names = ["schoening", "VCG - LLL_DM"]
_ = load_model_and_test_moser2(data_path, model_paths_list, model_names, 1000000, 2)
"""
