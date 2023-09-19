"""Test data_utils functions."""
import sys
import pytest
from allpairspy import AllPairs
from python.src.data_utils import SATTrainingDataset, JraphDataLoader
from python.src.sat_representations import VCG, LCG, SATRepresentation


sys.path.append("../../")


pairs = list(
    AllPairs(
        [
            [
                "python/tests/test_instances/single_instance/",
                "python/tests/test_instances/multiple_instances/",
            ],
            [VCG, LCG],
            [True, False],
            [1, 2],
            [True, False],
        ]
    )
)


class TestParameterized:
    """Test data utils functions."""

    @pytest.mark.parametrize(
        [
            "data_dir",
            "representation",
            "return_candidates",
            "batch_size",
            "include_constraint_graph",
        ],
        pairs,
    )
    def test_instance_loading(
        self,
        data_dir,
        representation,
        return_candidates,
        batch_size,
        include_constraint_graph,
    ):
        """Test instance loading."""
        assert batch_size > 0
        assert instance_loading_tester(
            data_dir, representation, return_candidates, include_constraint_graph
        )

    @pytest.mark.parametrize(
        [
            "data_dir",
            "representation",
            "return_candidates",
            "batch_size",
            "include_constraint_graph",
        ],
        pairs,
    )
    def test_collate_function(
        self,
        data_dir,
        representation,
        return_candidates,
        batch_size,
        include_constraint_graph,
    ):
        """Test collate function."""
        assert collate_function_tester(
            data_dir,
            representation,
            return_candidates,
            batch_size,
            include_constraint_graph,
        )


def instance_loading_tester(
    path: str,
    rep: SATRepresentation,
    return_candidates: bool = False,
    include_constraint_graph: bool = False,
):
    """Test instance loading."""
    dataset = SATTrainingDataset(
        data_dir=path,
        representation=rep,
        return_candidates=return_candidates,
        include_constraint_graph=include_constraint_graph,
    )
    max_nodes = dataset.max_n_node
    max_edges = dataset.max_n_edge
    for problem, (padded_candidates, energies) in dataset:
        graph = problem.graph
        n_node_array = graph.n_node
        assert n_node_array.sum() == max_nodes
        if rep == VCG:
            assert padded_candidates.shape[1] == max_nodes
        if rep == LCG:
            assert padded_candidates.shape[1] == max_nodes / 2
        assert energies.shape[0] == padded_candidates.shape[0]
        assert len(graph.receivers) == max_edges
        assert len(graph.senders) == max_edges
        assert len(graph.edges) == max_edges
        assert len(graph.nodes) == max_nodes
    return True


def collate_function_tester(
    path,
    rep: SATRepresentation,
    return_candidates: bool = False,
    batch_size=1,
    include_constraint_graph: bool = False,
):
    """Test collate function."""
    dataset = SATTrainingDataset(
        data_dir=path,
        representation=rep,
        return_candidates=return_candidates,
        include_constraint_graph=include_constraint_graph,
    )
    loader = JraphDataLoader(dataset, batch_size=batch_size)

    for i, batch in enumerate(loader):
        (masks, graphs, constraint_graphs, constraint_masks), (
            candidates,
            energies,
        ) = batch
        print("loading batch")
        batch_factor = (
            batch_size
            if (i + 1) * batch_size < len(dataset)
            else len(dataset) - i * batch_size
        )
        assert len(masks) == batch_factor * dataset.max_n_node
        assert graphs.n_node.sum() == batch_factor * dataset.max_n_node
        assert graphs.n_edge.sum() == batch_factor * dataset.max_n_edge
        assert len(graphs.receivers) == batch_factor * dataset.max_n_edge
        assert len(graphs.senders) == batch_factor * dataset.max_n_edge
        assert len(graphs.edges) == batch_factor * dataset.max_n_edge
        assert len(graphs.nodes) == batch_factor * dataset.max_n_node
        assert candidates.shape == energies.shape
        if include_constraint_graph:
            assert constraint_graphs.n_node.sum() == batch_factor * dataset.max_n_node
            assert constraint_masks.shape == masks.shape
        else:
            assert constraint_graphs is None
            assert constraint_masks is None
        if rep == VCG:
            assert len(candidates) == batch_factor * dataset.max_n_node
        elif rep == LCG:
            assert len(candidates) == batch_factor * dataset.max_n_node / 2
    return True
