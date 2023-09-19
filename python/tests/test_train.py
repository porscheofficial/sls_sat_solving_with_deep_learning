"""Test training loop."""
import sys
import pytest
from allpairspy import AllPairs
from python.src.train import train


sys.path.append("../../")


pairs = list(
    AllPairs(
        [
            [
                "python/tests/test_instances/single_instance/",
                "python/tests/test_instances/multiple_instances/",
            ],
            ["VCG", "LCG"],
            [True, False],
            [1, 2],
            [0, 1],
            [0],
            [0, 1],
            ["interaction", "GCN"],
        ]
    )
)


class TestParameterized:
    """Test training."""

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
        """Test training loop with different hyperparameters."""
        num_epochs = 1
        n_steps_moser = 0
        n_runs_moser = 1
        inv_temp = 1
        mlp_layers = [32, 32]
        train(
            batch_size=batch_size,
            inv_temp=inv_temp,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            num_epochs=num_epochs,
            n_steps_moser=n_steps_moser,
            n_runs_moser=n_runs_moser,
            data_path=data_dir,
            graph_representation=representation,
            network_type=network_type,
            return_candidates=return_candidates,
            mlp_layers=mlp_layers,
            initial_learning_rate=1e-3,
            final_learning_rate=1e-3,
        )
