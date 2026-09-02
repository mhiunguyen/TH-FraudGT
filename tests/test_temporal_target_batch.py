import pytest
import torch
from torch_geometric.data import HeteroData

from fraudGT.sampler.custom_sampler import PrepareTemporalLinkBatch


TASK = ('node', 'to', 'node')
REVERSE = ('node', 'rev_to', 'node')


def make_source():
    data = HeteroData()
    data['node'].x = torch.ones(4, 1)
    data['node'].num_nodes = 4
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]])
    edge_attr = torch.arange(6, dtype=torch.float32).view(3, 2)
    timestamps = torch.tensor([1, 2, 3], dtype=torch.long)
    for edge_type, index in [(TASK, edge_index),
                             (REVERSE, edge_index.flip(0))]:
        data[edge_type].edge_index = index
        data[edge_type].edge_attr = edge_attr.clone()
        data[edge_type].timestamps = timestamps.clone()
        data[edge_type].temporal_timestamps = timestamps.clone()
    data[TASK].y = torch.tensor([0, 1, 0])
    return data


def make_batch(existing_eid=0):
    batch = HeteroData()
    batch['node'].x = torch.ones(3, 1)
    batch['node'].num_nodes = 3
    batch['node'].batch = torch.zeros(3, dtype=torch.long)
    for edge_type, index in [
            (TASK, torch.tensor([[0], [1]])),
            (REVERSE, torch.tensor([[1], [0]]))]:
        batch[edge_type].edge_index = index
        batch[edge_type].edge_attr = torch.zeros(1, 2)
        batch[edge_type].e_id = torch.tensor([existing_eid])
        batch[edge_type].timestamps = torch.tensor([1])
        batch[edge_type].temporal_timestamps = torch.tensor([1])
    batch[TASK].y = torch.tensor([0])
    batch[TASK].edge_label_index = torch.tensor([[1], [2]])
    batch[TASK].edge_label = torch.tensor([1])
    batch[TASK].input_id = torch.tensor([0])
    return batch


def make_two_component_batch():
    """Target 1 is valid history for later target 2, not leakage."""
    batch = HeteroData()
    batch['node'].x = torch.ones(6, 1)
    batch['node'].num_nodes = 6
    batch['node'].batch = torch.tensor([0, 0, 0, 1, 1, 1])
    for edge_type, index in [
            (TASK, torch.tensor([[0, 3], [1, 4]])),
            (REVERSE, torch.tensor([[1, 4], [0, 3]]))]:
        batch[edge_type].edge_index = index
        batch[edge_type].edge_attr = torch.zeros(2, 2)
        batch[edge_type].e_id = torch.tensor([0, 1])
        batch[edge_type].timestamps = torch.tensor([1, 2])
        batch[edge_type].temporal_timestamps = torch.tensor([1, 2])
    batch[TASK].y = torch.tensor([0, 1])
    batch[TASK].edge_label_index = torch.tensor([[1, 4], [2, 5]])
    batch[TASK].edge_label = torch.tensor([1, 0])
    batch[TASK].input_id = torch.tensor([0, 1])
    return batch


def test_current_target_is_appended_and_marked():
    source = make_source()
    transform = PrepareTemporalLinkBatch(
        source, TASK, target_edge_ids=torch.tensor([1, 2]),
        add_ego_id=True)
    batch = transform(make_batch())

    assert batch[TASK].target_edge_mask.tolist() == [False, True]
    assert batch[TASK].e_id.tolist() == [0, 1]
    assert batch[TASK].y[batch[TASK].target_edge_mask].tolist() == [1]
    assert batch[REVERSE].e_id.tolist() == [0, 1]
    assert batch[TASK].edge_index[:, -1].tolist() == [1, 2]
    assert batch[REVERSE].edge_index[:, -1].tolist() == [2, 1]
    assert batch['node'].x[:, -1].tolist() == [0.0, 1.0, 1.0]


def test_target_already_in_history_is_rejected():
    source = make_source()
    transform = PrepareTemporalLinkBatch(
        source, TASK, target_edge_ids=torch.tensor([1]),
        add_ego_id=False)
    with pytest.raises(RuntimeError, match='Temporal leakage'):
        transform(make_batch(existing_eid=1))


def test_earlier_target_can_be_history_of_later_component():
    source = make_source()
    transform = PrepareTemporalLinkBatch(
        source, TASK, target_edge_ids=torch.tensor([1, 2]),
        add_ego_id=False)
    batch = transform(make_two_component_batch())

    assert batch[TASK].target_edge_mask.tolist() == [
        False, False, True, True]
    assert batch[TASK].e_id.tolist() == [0, 1, 1, 2]
    assert batch[TASK].y[batch[TASK].target_edge_mask].tolist() == [1, 0]
