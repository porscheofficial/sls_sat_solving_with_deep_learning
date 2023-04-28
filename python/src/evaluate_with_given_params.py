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
# data_path = "../Data/blocksworld"
data_path = "/Users/p403830/Library/CloudStorage/OneDrive-PorscheDigitalGmbH/programming/generateSAT/samples_n300"
# model_path = "../params_save/LCG_samples_large_n20230426-143758.npy"
model_path = "../params_save/LCG_blocksworld20230427-170155.npy"

N_STEPS_MOSER_list = [1, 10, 50, 100, 500, 1000, 5000, 10000, 50000, 100000, 500000]
N_RUNS_MOSER = 5
SEED = 0


def load_model_and_test_moser(data_path, model_path, N_STEPS_MOSER_list, N_RUNS_MOSER):
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

    energies_array = np.zeros(len(N_STEPS_MOSER_list))
    energies_array_uniform = np.zeros(len(N_STEPS_MOSER_list))
    for idx in range(len(sat_data)):
        problem_path = sat_data.instances[idx].name + ".cnf"
        problem = sat_data.get_unpadded_problem(idx)
        model_probabilities = get_model_probabilities(
            network, params, problem, graph_representation
        )
        model_probabilities = model_probabilities.ravel()
        uniform_probabilities = np.ones(len(model_probabilities)) / 2
        # print(np.max(model_probabilities), np.min(model_probabilities))
        single_energy = np.zeros(len(N_STEPS_MOSER_list))
        single_energy_uniform = np.zeros(len(N_STEPS_MOSER_list))
        for i, N_STEPS_MOSER in enumerate(N_STEPS_MOSER_list):
            _, _, final_energies = moser_rust.run_moser_python(
                problem_path, model_probabilities, N_STEPS_MOSER, N_RUNS_MOSER, SEED
            )
            single_energy[i] = final_energies
            _, _, final_energies_uniform = moser_rust.run_moser_python(
                problem_path, uniform_probabilities, N_STEPS_MOSER, N_RUNS_MOSER, SEED
            )
            single_energy_uniform[i] = final_energies_uniform
            # print("N_STEPS_MOSER", N_STEPS_MOSER, final_energies)
        # print(single_energy)
        energies_array += single_energy / problem.params[1]
        energies_array_uniform += single_energy_uniform / problem.params[1]

    energies_array = energies_array / len(sat_data)
    energies_array_uniform = energies_array_uniform / len(sat_data)
    plt.plot(N_STEPS_MOSER_list, energies_array, "--o", label="model")
    plt.plot(N_STEPS_MOSER_list, energies_array_uniform, "--o", label="uniform")
    plt.xlabel("N_STEPS_MOSER")
    plt.ylabel("# violated clauses / m")
    plt.xscale("log")
    plt.legend()
    plt.show()


load_model_and_test_moser(data_path, model_path, N_STEPS_MOSER_list, N_RUNS_MOSER)
