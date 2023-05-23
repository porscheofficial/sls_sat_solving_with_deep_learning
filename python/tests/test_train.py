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
            [0, 1],
            [0, 1],
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
    ):
        NUM_EPOCHS = 1
        N_STEPS_MOSER = 100
        N_RUNS_MOSER = 1
        f = 0.0000001
        train(
            batch_size=batch_size,
            inv_temp=f,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            NUM_EPOCHS=NUM_EPOCHS,
            N_STEPS_MOSER=N_STEPS_MOSER,
            N_RUNS_MOSER=N_RUNS_MOSER,
            path=data_dir,
            graph_representation=representation,
            return_candidates=return_candidates,
        )
