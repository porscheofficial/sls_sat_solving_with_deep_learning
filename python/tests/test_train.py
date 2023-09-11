import pytest
import sys
from allpairspy import AllPairs

sys.path.append("../../")

from python.src.data_utils import SATTrainingDataset, JraphDataLoader
from python.src.sat_representations import VCG, LCG, SATRepresentation
from python.src.model import (
    get_network_definition,
)
from python.src.train import train

pairs = [
    values
    for values in AllPairs(
        [
            [
                "python/tests/test_instances/single_instance/",
                "python/tests/test_instances/multiple_instances/",
            ],
            [VCG, LCG],
            [True, False],
            [1, 2],
            [0, 1],
            [0],
            [0, 1],
            ["interaction", "GCN"],
        ]
    )
]


class TestParameterized(object):
    @pytest.mark.parametrize(
        [
            "data_dir",
            "representation",
            "return_candidates",
            "batch_size",
            "alpha",
            "beta",
            "gamma",
            "network_type",
        ],
        pairs,
    )
    def test_train(
        self,
        data_dir,
        representation,
        return_candidates,
        batch_size,
        alpha,
        beta,
        gamma,
        network_type,
    ):
        NUM_EPOCHS = 1
        N_STEPS_MOSER = 0
        N_RUNS_MOSER = 1
        inv_temp = 1
        mlp_layers = [32, 32]
        train(
            batch_size=batch_size,
            inv_temp=inv_temp,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            num_epochs=NUM_EPOCHS,
            n_steps_moser=N_STEPS_MOSER,
            n_runs_moser=N_RUNS_MOSER,
            data_path=data_dir,
            graph_representation_rep=representation,
            network_type=network_type,
            return_candidates=return_candidates,
            mlp_layers=mlp_layers,
            initial_learning_rate=1e-3,
            final_learning_rate=1e-3,
        )
