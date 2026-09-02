import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index

from fraudGT.graphgym.register import register_head
from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.models.layer import MLP


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
        else:
            # There can be parallel edges between one node pair, so edge IDs
            # are the safest key for the original non-temporal loader.
            mask = torch.isin(
                batch[task].e_id,
                getattr(self, f'{batch.split}_inds')[batch[task].input_id])

        if not mask.any():
            raise RuntimeError(
                f'No supervision edges found in {batch.split} batch.')

        task = cfg.dataset.task_entity
        edge_index = batch[task].edge_index

        # A concatentation of source/target node embedding + edge attribute
        features = torch.cat(
            (batch[task[0]].x[edge_index[0, mask]],
             batch[task[2]].x[edge_index[1, mask]],
             batch[task].edge_attr[mask]),
            dim=-1)
        labels = batch[task].y[mask]
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
