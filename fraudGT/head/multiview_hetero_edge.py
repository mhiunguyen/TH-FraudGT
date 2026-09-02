"""Multi-view gated edge-classification head for MV-IA-FraudGT."""

import torch
import torch.nn as nn
from torch_geometric.data import HeteroData
from torch_geometric.utils import mask_to_index

from fraudGT.graphgym.config import cfg
from fraudGT.graphgym.register import register_head


def _mlp(dim_in, dim_hidden, dim_out, num_layers, dropout):
    """Build a small GELU MLP without depending on GraphGym global defaults."""
    if num_layers < 1:
        raise ValueError('num_layers must be at least one')

    layers = []
    current = dim_in
    for _ in range(num_layers - 1):
        layers.extend([
            nn.Linear(current, dim_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        current = dim_hidden
    layers.append(nn.Linear(current, dim_out))
    return nn.Sequential(*layers)


@register_head('multiview_hetero_edge')
class MultiViewHeteroEdgeHead(nn.Module):
    """Fuse FraudGT graph context with a direct transaction-feature view.

    Graph view:
        MLP([h_src || h_dst || e_final])
    Transaction view:
        MLP(e_input)
    Fusion:
        alpha * z_graph + (1 - alpha) * z_transaction

    ``alpha`` is learned independently for each target transaction.
    """

    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        if not isinstance(dataset[0], HeteroData):
            raise TypeError('MultiViewHeteroEdgeHead requires HeteroData')

        self.task = cfg.dataset.task_entity
        self.train_inds = mask_to_index(
            dataset['train'][self.task].split_mask).to(cfg.device)
        self.val_inds = mask_to_index(
            dataset['val'][self.task].split_mask).to(cfg.device)
        self.test_inds = mask_to_index(
            dataset['test'][self.task].split_mask).to(cfg.device)

        raw_dim = dataset['train'][self.task].edge_attr.size(-1)
        hidden = cfg.mvia.dim_hidden
        dropout = cfg.mvia.dropout

        self.graph_projector = _mlp(
            dim_in * 3,
            hidden,
            hidden,
            cfg.mvia.graph_layers,
            dropout,
        )
        self.transaction_projector = _mlp(
            raw_dim,
            hidden,
            hidden,
            cfg.mvia.transaction_layers,
            dropout,
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden * 2, cfg.mvia.gate_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(cfg.mvia.gate_hidden, 1),
            nn.Sigmoid(),
        )
        self.classifier = _mlp(
            hidden,
            hidden,
            dim_out,
            cfg.mvia.classifier_layers,
            dropout,
        )

        # Exposed for diagnostics/explainability after a forward pass.
        self.last_gate = None

    def _target_mask(self, batch):
        if hasattr(batch[self.task], 'target_edge_mask'):
            mask = batch[self.task].target_edge_mask
            expected = batch[self.task].edge_label.numel()
            if int(mask.sum()) != expected:
                raise RuntimeError(
                    'Temporal target mask does not match edge supervision.')
            return mask
        split_inds = getattr(self, f'{batch.split}_inds')
        return torch.isin(
            batch[self.task].e_id,
            split_inds[batch[self.task].input_id],
        )

    def forward(self, batch):
        edge_store = batch[self.task]
        if not hasattr(edge_store, 'raw_edge_attr'):
            raise RuntimeError(
                'raw_edge_attr is missing. Use GTModel with '
                "gt.head='multiview_hetero_edge'."
            )

        mask = self._target_mask(batch)
        edge_index = edge_store.edge_index
        src, _, dst = self.task

        graph_input = torch.cat([
            batch[src].x[edge_index[0, mask]],
            batch[dst].x[edge_index[1, mask]],
            edge_store.edge_attr[mask],
        ], dim=-1)
        transaction_input = edge_store.raw_edge_attr[mask]

        z_graph = self.graph_projector(graph_input)
        z_transaction = self.transaction_projector(transaction_input)
        alpha = self.gate(torch.cat([z_graph, z_transaction], dim=-1))
        z_fusion = alpha * z_graph + (1.0 - alpha) * z_transaction

        self.last_gate = alpha.detach()
        prediction = self.classifier(z_fusion)
        label = edge_store.y[mask]
        return prediction, label
