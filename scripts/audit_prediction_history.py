"""Prediction-level audit of FraudGT sampling coverage.

The audit loads a trained strict-past checkpoint, performs deterministic
full-split inference, and joins every target prediction with exact direct
history and sampled-history counts.  It is intentionally descriptive: the
output can support or reject a history-coverage hypothesis, but it does not
claim that low coverage causes an error.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch_geometric import seed_everything
from torch_geometric.utils import mask_to_index
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fraudGT  # noqa: E402,F401  (register project components)
from fraudGT.graphgym.config import cfg, load_cfg, set_cfg  # noqa: E402
from fraudGT.graphgym.loader import create_dataset, get_loader  # noqa: E402
from fraudGT.graphgym.model_builder import create_model  # noqa: E402
from history_count_utils import exact_history_counts  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', required=True, type=Path)
    parser.add_argument('--run-dir', required=True, type=Path)
    parser.add_argument('--selection-summary', required=True, type=Path)
    parser.add_argument('--output-dir', required=True, type=Path)
    parser.add_argument('--split', choices=['val', 'test'], default='val')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def _safe_metric(func, y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return math.nan
    return float(func(y_true, y_pred, zero_division=0))


def classification_metrics(
    frame: pd.DataFrame,
    threshold: float,
) -> dict[str, float | int]:
    y_true = frame['label'].to_numpy(dtype=np.int64)
    scores = frame['score'].to_numpy(dtype=np.float64)
    y_pred = (scores > threshold).astype(np.int64)
    result: dict[str, float | int] = {
        'n': int(len(frame)),
        'fraud': int(y_true.sum()),
        'fraud_rate': float(y_true.mean()) if len(y_true) else math.nan,
        'predicted_fraud': int(y_pred.sum()),
        'precision': _safe_metric(precision_score, y_true, y_pred),
        'recall': _safe_metric(recall_score, y_true, y_pred),
        'f1': _safe_metric(f1_score, y_true, y_pred),
        'error_rate': float((y_true != y_pred).mean()) if len(y_true) else math.nan,
    }
    if len(np.unique(y_true)) == 2:
        result['ap'] = float(average_precision_score(y_true, scores))
        result['auc'] = float(roc_auc_score(y_true, scores))
    else:
        result['ap'] = math.nan
        result['auc'] = math.nan
    return result


def grouped_metrics(
    frame: pd.DataFrame,
    group_column: str,
    threshold: float,
) -> pd.DataFrame:
    rows = []
    for name, group in frame.groupby(group_column, observed=False, sort=False):
        row = {group_column: str(name)}
        row.update(classification_metrics(group, threshold))
        row['history_median'] = float(group['full_history_count'].median())
        row['coverage_mean'] = float(group['history_coverage'].mean())
        row['coverage_median'] = float(group['history_coverage'].median())
        rows.append(row)
    return pd.DataFrame(rows)


def batch_sample_statistics(
    batch,
    task,
    global_src: torch.Tensor,
    global_dst: torch.Tensor,
    global_times: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Count unique directly relevant sampled edges per target component."""
    store = batch[task]
    target_eids = store.target_e_id.to(dtype=torch.long, device='cpu')
    target_index = store.edge_label_index.to(device='cpu')
    node_batch = batch[task[0]].batch.to(dtype=torch.long, device='cpu')
    target_groups = node_batch[target_index[0]]
    destination_groups = node_batch[target_index[1]]
    if not torch.equal(target_groups, destination_groups):
        raise RuntimeError('Target endpoints are in different components')
    if target_groups.unique().numel() != target_eids.numel():
        raise RuntimeError('Expected one disjoint component per target')

    group_count = int(node_batch.max()) + 1
    target_by_group = torch.full((group_count,), -1, dtype=torch.long)
    target_by_group[target_groups] = target_eids
    encoded = []
    edge_count = int(global_src.numel())
    for relation in batch.edge_types:
        relation_store = batch[relation]
        if not hasattr(relation_store, 'e_id'):
            continue
        eids = relation_store.e_id.to(dtype=torch.long, device='cpu')
        groups = node_batch[relation_store.edge_index[0].to(device='cpu')]
        keep = eids != target_by_group[groups]
        if keep.any():
            encoded.append(groups[keep] * edge_count + eids[keep])
    if encoded:
        unique_keys = torch.unique(torch.cat(encoded))
        sampled_groups = torch.div(unique_keys, edge_count, rounding_mode='floor')
        sampled_eids = unique_keys.remainder(edge_count)
    else:
        sampled_groups = torch.empty(0, dtype=torch.long)
        sampled_eids = torch.empty(0, dtype=torch.long)

    target_src_by_group = torch.full((group_count,), -1, dtype=torch.long)
    target_dst_by_group = torch.full((group_count,), -1, dtype=torch.long)
    target_time_by_group = torch.full((group_count,), -1, dtype=torch.long)
    target_src_by_group[target_groups] = global_src[target_eids]
    target_dst_by_group[target_groups] = global_dst[target_eids]
    target_time_by_group[target_groups] = global_times[target_eids]

    if sampled_eids.numel():
        sampled_src = global_src[sampled_eids]
        sampled_dst = global_dst[sampled_eids]
        incident = (
            (sampled_src == target_src_by_group[sampled_groups])
            | (sampled_dst == target_src_by_group[sampled_groups])
            | (sampled_src == target_dst_by_group[sampled_groups])
            | (sampled_dst == target_dst_by_group[sampled_groups])
        )
        same_pair = (
            (sampled_src == target_src_by_group[sampled_groups])
            & (sampled_dst == target_dst_by_group[sampled_groups])
        )
        sampled_times = global_times[sampled_eids]
        if not torch.all(sampled_times < target_time_by_group[sampled_groups]):
            raise RuntimeError('Strict-past audit failed during inference')
        direct_by_group = torch.bincount(
            sampled_groups[incident], minlength=group_count)
        pair_by_group = torch.bincount(
            sampled_groups[same_pair], minlength=group_count)
        unique_by_group = torch.bincount(
            sampled_groups, minlength=group_count)
    else:
        direct_by_group = torch.zeros(group_count, dtype=torch.long)
        pair_by_group = torch.zeros(group_count, dtype=torch.long)
        unique_by_group = torch.zeros(group_count, dtype=torch.long)

    return (
        target_eids,
        direct_by_group[target_groups],
        pair_by_group[target_groups],
        unique_by_group[target_groups],
    )


def configure(config_path: Path, device: str, seed: int) -> None:
    set_cfg(cfg)
    load_cfg(cfg, SimpleNamespace(cfg_file=str(config_path), opts=[]))
    cfg.device = device
    cfg.seed = seed
    cfg.run_id = seed
    torch.set_num_threads(cfg.num_threads)
    seed_everything(seed)


def load_selection(summary_path: Path, seed: int) -> tuple[int, float, pd.Series]:
    summary = pd.read_csv(summary_path)
    selected = summary.loc[summary['seed'].astype(int) == seed]
    if len(selected) != 1:
        raise RuntimeError(f'Expected one selection row for seed {seed}')
    row = selected.iloc[0]
    return int(row['best_epoch']), float(row['threshold']), row


def run_audit(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure(args.cfg, args.device, args.seed)
    best_epoch, threshold, expected = load_selection(
        args.selection_summary, args.seed)
    print(f'Selected only by validation: epoch={best_epoch}, threshold={threshold:.2f}')

    dataset = create_dataset()
    task = cfg.dataset.task_entity
    split_data = dataset[args.split]
    loader = get_loader(
        dataset,
        cfg.val.sampler,
        cfg.train.batch_size,
        shuffle=False,
        split=args.split,
    )
    loader.set_step(-1)

    checkpoint = args.run_dir / 'ckpt' / f'{best_epoch}.ckpt'
    if not checkpoint.exists():
        available = sorted(path.name for path in (args.run_dir / 'ckpt').glob('*.ckpt'))
        raise FileNotFoundError(
            f'Missing selected checkpoint {checkpoint}; available={available}')
    model = create_model(dataset=dataset)
    state = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(state['model_state'])
    model.eval()

    store = split_data[task]
    global_src = store.edge_index[0].to(dtype=torch.long, device='cpu')
    global_dst = store.edge_index[1].to(dtype=torch.long, device='cpu')
    global_times = store.timestamps.round().to(dtype=torch.long, device='cpu')
    global_labels = store.y.to(dtype=torch.long, device='cpu')

    edge_ids, labels, scores = [], [], []
    sampled_history, sampled_pair, sampled_unique = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Prediction audit ({args.split})'):
            target_eids, direct_count, pair_count, unique_count = \
                batch_sample_statistics(
                    batch, task, global_src, global_dst, global_times)
            batch.split = args.split
            batch.to(torch.device(args.device))
            logits, true = model(batch)
            score = torch.sigmoid(logits.reshape(-1)).detach().cpu()
            true = true.reshape(-1).detach().cpu().to(dtype=torch.long)
            if not torch.equal(true, global_labels[target_eids]):
                raise RuntimeError('Prediction labels do not match target edge IDs')
            if score.numel() != target_eids.numel():
                raise RuntimeError('Prediction count does not match target edge IDs')
            edge_ids.append(target_eids)
            labels.append(true)
            scores.append(score)
            sampled_history.append(direct_count)
            sampled_pair.append(pair_count)
            sampled_unique.append(unique_count)

    target_eids = torch.cat(edge_ids).numpy().astype(np.int64, copy=False)
    labels_np = torch.cat(labels).numpy().astype(np.int8, copy=False)
    scores_np = torch.cat(scores).numpy().astype(np.float64, copy=False)
    sampled_history_np = torch.cat(sampled_history).numpy().astype(np.int64, copy=False)
    sampled_pair_np = torch.cat(sampled_pair).numpy().astype(np.int64, copy=False)
    sampled_unique_np = torch.cat(sampled_unique).numpy().astype(np.int64, copy=False)
    expected_ids = mask_to_index(store.split_mask).cpu().numpy()
    if not np.array_equal(np.sort(target_eids), expected_ids):
        raise RuntimeError('Inference did not cover every split target exactly once')

    # Keep only the audited cumulative graph before the memory-heavy exact
    # history sort. The other two split graphs and the GPU model are no longer
    # needed once predictions have been collected.
    del loader, model
    for other_split in ('train', 'val', 'test'):
        if other_split != args.split:
            dataset.data_dict[other_split] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('Computing exact strict-past history counts...')
    src_np = global_src.numpy()
    dst_np = global_dst.numpy()
    times_np = global_times.numpy()
    full_history, full_pair = exact_history_counts(
        src_np, dst_np, times_np, target_eids, split_data['node'].num_nodes)
    if np.any(sampled_history_np > full_history):
        raise RuntimeError('Sampled direct history exceeds full direct history')
    if np.any(sampled_pair_np > full_pair):
        raise RuntimeError('Sampled pair history exceeds full pair history')

    history_coverage = np.divide(
        sampled_history_np,
        full_history,
        out=np.full(len(full_history), np.nan, dtype=np.float64),
        where=full_history > 0,
    )
    pair_coverage = np.divide(
        sampled_pair_np,
        full_pair,
        out=np.full(len(full_pair), np.nan, dtype=np.float64),
        where=full_pair > 0,
    )
    prediction = (scores_np > threshold).astype(np.int8)
    frame = pd.DataFrame({
        'split': args.split,
        'seed': args.seed,
        'target_eid': target_eids,
        'timestamp': times_np[target_eids],
        'source': src_np[target_eids],
        'target': dst_np[target_eids],
        'label': labels_np,
        'score': scores_np,
        'threshold': threshold,
        'prediction': prediction,
        'error': (prediction != labels_np).astype(np.int8),
        'full_history_count': full_history,
        'sampled_history_count': sampled_history_np,
        'history_coverage': history_coverage,
        'full_pair_count': full_pair,
        'sampled_pair_count': sampled_pair_np,
        'pair_history_coverage': pair_coverage,
        'sampled_unique_edges': sampled_unique_np,
    }).sort_values('target_eid').reset_index(drop=True)

    history_bins = [-0.5, 0.5, 10.5, 100.5, 1000.5, 10000.5, np.inf]
    history_labels = ['0', '1–10', '11–100', '101–1,000', '1,001–10,000', '>10,000']
    frame['history_group'] = pd.cut(
        frame['full_history_count'], history_bins,
        labels=history_labels, include_lowest=True, ordered=True)
    coverage_group = pd.cut(
        frame['history_coverage'],
        [-1e-12, 0.25, 0.50, 0.75, 1.0],
        labels=['[0, 0.25]', '(0.25, 0.50]', '(0.50, 0.75]', '(0.75, 1.00]'],
        include_lowest=True, ordered=True)
    frame['coverage_group'] = coverage_group.astype(object)
    frame.loc[frame['full_history_count'] == 0, 'coverage_group'] = 'no history'
    coverage_order = ['no history', '[0, 0.25]', '(0.25, 0.50]', '(0.50, 0.75]', '(0.75, 1.00]']
    frame['coverage_group'] = pd.Categorical(
        frame['coverage_group'], categories=coverage_order, ordered=True)

    global_result = classification_metrics(frame, threshold)
    expected_metric = 'val_f1' if args.split == 'val' else 'test_f1'
    if abs(float(global_result['f1']) - float(expected[expected_metric])) > 5e-4:
        raise RuntimeError(
            f'Inference F1 {global_result["f1"]:.5f} does not reproduce '
            f'{expected_metric}={float(expected[expected_metric]):.5f}')
    global_result.update({
        'split': args.split,
        'seed': args.seed,
        'best_epoch': best_epoch,
        'threshold': threshold,
        'checkpoint': str(checkpoint),
    })

    detailed_path = args.output_dir / f'prediction_history_audit_{args.split}_seed{args.seed}.csv'
    history_path = args.output_dir / f'prediction_by_history_group_{args.split}_seed{args.seed}.csv'
    coverage_path = args.output_dir / f'prediction_by_coverage_group_{args.split}_seed{args.seed}.csv'
    global_path = args.output_dir / f'prediction_audit_global_{args.split}_seed{args.seed}.json'
    figure_path = args.output_dir / f'prediction_history_audit_{args.split}_seed{args.seed}.png'
    frame.to_csv(detailed_path, index=False)
    history_summary = grouped_metrics(frame, 'history_group', threshold)
    coverage_summary = grouped_metrics(frame, 'coverage_group', threshold)
    history_summary.to_csv(history_path, index=False)
    coverage_summary.to_csv(coverage_path, index=False)
    global_path.write_text(json.dumps(global_result, indent=2), encoding='utf-8')

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    history_summary.plot(
        x='history_group', y='coverage_mean', kind='bar', legend=False,
        color='#4472C4', ax=axes[0])
    axes[0].set_title('Mean sampling coverage by history size')
    axes[0].set_xlabel('Strict-past direct history'); axes[0].set_ylabel('Coverage')
    history_summary.plot(
        x='history_group', y=['recall', 'f1'], kind='bar',
        color=['#ED7D31', '#70AD47'], ax=axes[1])
    axes[1].set_title('Prediction quality by history size')
    axes[1].set_xlabel('Strict-past direct history'); axes[1].set_ylabel('Score')
    coverage_summary.plot(
        x='coverage_group', y=['recall', 'f1'], kind='bar',
        color=['#ED7D31', '#70AD47'], ax=axes[2])
    axes[2].set_title('Prediction quality by sampling coverage')
    axes[2].set_xlabel('Coverage'); axes[2].set_ylabel('Score')
    for axis in axes:
        axis.tick_params(axis='x', rotation=25)
        axis.grid(axis='y', alpha=.25)
    fig.suptitle(
        f'Notebook 15 — prediction/history audit ({args.split}, seed {args.seed})')
    fig.tight_layout()
    fig.savefig(figure_path, dpi=180, bbox_inches='tight')
    plt.close(fig)

    print('\nGlobal reproduction:')
    print(json.dumps(global_result, indent=2))
    print('\nBy history size:')
    print(history_summary.to_string(index=False))
    print('\nBy coverage:')
    print(coverage_summary.to_string(index=False))
    print(f'\nSaved audit files to {args.output_dir}')


def main() -> None:
    run_audit(parse_args())


if __name__ == '__main__':
    main()
