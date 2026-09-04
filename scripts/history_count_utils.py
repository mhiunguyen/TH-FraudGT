"""Memory-conscious strict-past history counting utilities."""

from __future__ import annotations

import numpy as np


def strict_prior_counts(keys: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Count prior records with the same key and a strictly smaller time."""
    keys = np.asarray(keys, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    if keys.ndim != 1 or times.ndim != 1 or len(keys) != len(times):
        raise ValueError('keys and times must be equally sized 1-D arrays')
    if len(keys) == 0:
        return np.empty(0, dtype=np.int64)

    order = np.lexsort((times, keys))
    sorted_keys = keys[order]
    sorted_times = times[order]
    positions = np.arange(len(order), dtype=np.int64)
    new_key = np.empty(len(order), dtype=bool)
    new_key[0] = True
    new_key[1:] = sorted_keys[1:] != sorted_keys[:-1]
    new_time = new_key.copy()
    new_time[1:] |= sorted_times[1:] != sorted_times[:-1]
    key_start = np.maximum.accumulate(np.where(new_key, positions, 0))
    time_start = np.maximum.accumulate(np.where(new_time, positions, 0))
    prior_sorted = time_start - key_start
    prior = np.empty_like(prior_sorted)
    prior[order] = prior_sorted
    return prior


def exact_history_counts(
    src: np.ndarray,
    dst: np.ndarray,
    times: np.ndarray,
    target_eids: np.ndarray,
    num_nodes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return exact strict-past incident-history and directed-pair counts."""
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    target_eids = np.asarray(target_eids, dtype=np.int64)
    if not (len(src) == len(dst) == len(times)):
        raise ValueError('src, dst and times must have equal lengths')

    non_self = src != dst
    endpoint_nodes = np.concatenate([src, dst[non_self]])
    endpoint_times = np.concatenate([times, times[non_self]])
    endpoint_prior = strict_prior_counts(endpoint_nodes, endpoint_times)
    src_prior = endpoint_prior[:len(src)]
    dst_prior = np.empty(len(src), dtype=np.int64)
    dst_prior[non_self] = endpoint_prior[len(src):]
    dst_prior[~non_self] = src_prior[~non_self]
    del endpoint_nodes, endpoint_times, endpoint_prior

    low = np.minimum(src, dst)
    high = np.maximum(src, dst)
    undirected_keys = low * np.int64(num_nodes) + high
    undirected_prior = strict_prior_counts(undirected_keys, times)
    directed_keys = src * np.int64(num_nodes) + dst
    directed_prior = strict_prior_counts(directed_keys, times)

    total_history = src_prior[target_eids].copy()
    target_non_self = non_self[target_eids]
    ns_ids = target_eids[target_non_self]
    total_history[target_non_self] = (
        src_prior[ns_ids] + dst_prior[ns_ids] - undirected_prior[ns_ids]
    )
    pair_history = directed_prior[target_eids]
    if (total_history < 0).any() or (pair_history < 0).any():
        raise RuntimeError('Negative history count indicates an audit bug')
    return total_history, pair_history
