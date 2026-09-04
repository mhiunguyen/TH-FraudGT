from fraudGT.graphgym.register import register_config
from typing import Union

@register_config('dataset_cfg')
def dataset_cfg(cfg):
    """Dataset-specific config options.
    """

    # The entity to perform the task in an heterogeneous graph dataset
    cfg.dataset.task_entity = None

    # The number of node types to expect in TypeDictNodeEncoder.
    cfg.dataset.node_encoder_num_types = 0

    # The number of edge types to expect in TypeDictEdgeEncoder.
    cfg.dataset.edge_encoder_num_types = 0

    # VOC/COCO Superpixels dataset version based on SLIC compactness parameter.
    cfg.dataset.slic_compactness = 10

    # infer-link parameters (e.g., edge prediction task)
    cfg.dataset.infer_link_label = "None"

    cfg.dataset.reverse_mp = False
    cfg.dataset.add_ports = False

    # Add leakage-safe historical behavior features to AML transaction edges.
    # This option uses a separate processed cache from the original dataset.
    cfg.dataset.add_history = False

    # Select an ablation subset from recency, frequency, and monetary.  The
    # endpoint_behavior preset selects source outgoing count, destination
    # incoming count, and source amount deviation.  The default preserves the
    # original H-FraudGT behavior with all 8 features.
    cfg.dataset.history_groups = ['recency', 'frequency', 'monetary']

    # Optional reliability shrinkage for H-FraudGT. Continuous historical
    # features are multiplied by n / (n + kappa), where n is the relevant
    # strictly-prior transaction count. The original H-RFM remains unchanged
    # when this switch is False.
    cfg.dataset.history_reliability = False
    cfg.dataset.history_reliability_kappa = 5.0

    cfg.dataset.rand_split = False
