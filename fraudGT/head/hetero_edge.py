import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


def ordered_edge_positions(sampled_eids, target_eids):
    """Map target edge IDs to sampled-edge positions in target order."""
    if target_eids.numel() == 0:
        raise RuntimeError('Edge batch contains no supervision targets.')
    if sampled_eids.numel() == 0:
        raise RuntimeError('Sampled batch contains no edges.')

    sorted_eids, permutation = torch.sort(sampled_eids)
    locations = torch.searchsorted(sorted_eids, target_eids)
    in_range = locations < sorted_eids.numel()
    safe_locations = locations.clamp(max=sorted_eids.numel() - 1)
    matched = in_range & (sorted_eids[safe_locations] == target_eids)
    if not matched.all():
        missing = target_eids[~matched][:10].detach().cpu().tolist()
        raise RuntimeError(
            'Target edge IDs are missing from the sampled batch: '
            f'{missing}.')

    positions = permutation[safe_locations]
    if positions.unique().numel() != target_eids.numel():
        raise RuntimeError('Target edge IDs do not map one-to-one to edges.')
    return positions


@register_head('hetero_edge')
class HeteroGNNEdgeHead(nn.Module):
    '''Head of Hetero GNN, edge prediction'''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        # self.train_edge_inds = mask_to_index(data[cfg.dataset.task_entity].train_edge_mask).to(cfg.device)
        # self.val_edge_inds = mask_to_index(data[cfg.dataset.task_entity].val_edge_mask).to(cfg.device)
        # self.test_edge_inds = mask_to_index(data[cfg.dataset.task_entity].test_edge_mask).to(cfg.device)
        self.train_inds = mask_to_index(dataset['train'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(dataset['val'][cfg.dataset.task_entity].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(dataset['test'][cfg.dataset.task_entity].split_mask).to(cfg.device)

        self.layer_post_mp = MLP(dim_in * 3, dim_out, 
                                 num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
                                 bias=True)
        # requires parameter
        # self.decode_module = lambda v1, v2: \
        #     self.layer_post_mp(torch.cat((v1, v2), dim=-1))

    def _apply_index(self, batch):
        task = cfg.dataset.task_entity
        if hasattr(batch[task], 'target_edge_mask'):
            mask = batch[task].target_edge_mask
            expected = batch[task].edge_label.numel()
            actual = int(mask.sum())
            if actual != expected:
                raise RuntimeError(
                    'Temporal target mask mismatch: '
                    f'expected {expected} targets, found {actual}.')
            positions = mask_to_index(mask)
        else:
            # Preserve the input supervision order. A boolean ``isin`` mask
            # returns sampled-adjacency order, which can differ from
            # ``edge_label`` order and trigger false label mismatches.
            target_eids = getattr(
                self, f'{batch.split}_inds')[batch[task].input_id]
            positions = ordered_edge_positions(
                batch[task].e_id, target_eids)

        if positions.numel() == 0:
            raise RuntimeError(
                f'No supervision edges found in {batch.split} batch.')

        task = cfg.dataset.task_entity
        edge_index = batch[task].edge_index

        # A concatentation of source/target node embedding + edge attribute
        features = torch.cat(
            (batch[task[0]].x[edge_index[0, positions]],
             batch[task[2]].x[edge_index[1, positions]],
             batch[task].edge_attr[positions]),
            dim=-1)
        labels = batch[task].y[positions]
        if hasattr(batch[task], 'edge_label') and \
                labels.numel() == batch[task].edge_label.numel() and \
                not torch.equal(
                    labels.view(-1), batch[task].edge_label.view(-1)):
            raise RuntimeError(
                'Prediction-head labels do not match edge supervision labels.')
        return features, labels
    

    def forward(self, batch):
        # TODO: add homogeneous graph support
        # batch.x_dict[cfg.dataset.task_entity] = self.layer_post_mp(batch.x_dict[cfg.dataset.task_entity])
        # pred, label = self._apply_index(batch)
    
        # if cfg.model.edge_decoding != 'concat':
        #     batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        # nodes_first = pred[0]
        # nodes_second = pred[1]
        # pred = self.decode_module(nodes_first, nodes_second)
        pred = self.layer_post_mp(pred)

        return pred, label
    
        # if not self.training:  # Compute extra stats when in evaluation mode.
        #     stats = self.compute_mrr(batch)
        #     return pred, label, stats
        # else:
        #     return pred, label
