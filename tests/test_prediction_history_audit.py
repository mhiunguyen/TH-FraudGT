import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / 'scripts' / \
    'history_count_utils.py'
SPEC = importlib.util.spec_from_file_location('history_count_utils', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_strict_prior_counts_excludes_same_timestamp():
    keys = np.array([1, 1, 1, 2, 1])
    times = np.array([10, 10, 11, 9, 12])
    actual = MODULE.strict_prior_counts(keys, times)
    assert actual.tolist() == [0, 0, 2, 0, 3]


def test_exact_history_counts_uses_unique_incident_edges():
    src = np.array([0, 0, 0, 1, 0])
    dst = np.array([1, 2, 1, 0, 0])
    times = np.array([1, 1, 2, 3, 3])
    target_eids = np.arange(len(src))

    full_history, directed_pair = MODULE.exact_history_counts(
        src, dst, times, target_eids, num_nodes=3)

    assert full_history.tolist() == [0, 0, 2, 3, 3]
    assert directed_pair.tolist() == [0, 0, 1, 0, 0]


def test_exact_history_counts_matches_brute_force():
    rng = np.random.default_rng(42)
    num_nodes = 12
    src = rng.integers(0, num_nodes, size=200)
    dst = rng.integers(0, num_nodes, size=200)
    times = np.sort(rng.integers(0, 20, size=200))
    target_eids = np.arange(len(src))

    full_history, directed_pair = MODULE.exact_history_counts(
        src, dst, times, target_eids, num_nodes=num_nodes)

    expected_history = []
    expected_pair = []
    for edge_id, (u, v, timestamp) in enumerate(zip(src, dst, times)):
        prior = np.flatnonzero(times < timestamp)
        incident = prior[
            (src[prior] == u) | (dst[prior] == u)
            | (src[prior] == v) | (dst[prior] == v)
        ]
        pair = prior[(src[prior] == u) & (dst[prior] == v)]
        expected_history.append(len(np.unique(incident)))
        expected_pair.append(len(pair))

    assert full_history.tolist() == expected_history
    assert directed_pair.tolist() == expected_pair
