import importlib.util
from pathlib import Path

import torch
from torch_geometric.data import HeteroData


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / \
    'audit_prediction_history.py'
SPEC = importlib.util.spec_from_file_location('prediction_history_audit', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_batch_statistics_deduplicates_relations_and_excludes_targets():
    task = ('node', 'to', 'node')
    reverse = ('node', 'rev_to', 'node')
    batch = HeteroData()
    batch['node'].batch = torch.tensor([0, 0, 1, 1, 0, 0])
    batch['node'].num_nodes = 6

    # Edges 0 and 2 are direct history. Edge 4 is a two-hop edge in group 0.
    # Edges 1 and 3 are the appended prediction targets.
    forward_index = torch.tensor([
        [0, 2, 4, 0, 2],
        [1, 3, 5, 1, 3],
    ])
    batch[task].edge_index = forward_index
    batch[task].e_id = torch.tensor([0, 2, 4, 1, 3])
    batch[task].edge_label_index = torch.tensor([[0, 2], [1, 3]])
    batch[task].target_e_id = torch.tensor([1, 3])
    batch[reverse].edge_index = forward_index.flip(0)
    batch[reverse].e_id = torch.tensor([0, 2, 4, 1, 3])

    global_src = torch.tensor([0, 0, 2, 2, 4])
    global_dst = torch.tensor([1, 1, 3, 3, 5])
    global_times = torch.tensor([1, 2, 1, 2, 1])

    target_eids, direct, pair, unique = MODULE.batch_sample_statistics(
        batch, task, global_src, global_dst, global_times)

    assert target_eids.tolist() == [1, 3]
    assert direct.tolist() == [1, 1]
    assert pair.tolist() == [1, 1]
    assert unique.tolist() == [2, 1]
