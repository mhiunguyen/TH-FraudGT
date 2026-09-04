import numpy as np
import pandas as pd

from fraudGT.datasets.history_features import (
    HISTORY_FEATURE_NAMES,
    apply_history_reliability_gate,
    compute_past_only_history_features_raw,
    history_feature_names,
    normalize_history_features_train_only,
    resolve_history_feature_indices,
)


def test_same_timestamp_edges_do_not_observe_each_other():
    frame = pd.DataFrame({
        'from_id': [0, 0, 0, 3],
        'to_id': [1, 2, 1, 1],
        'Timestamp': [10, 10, 20, 20],
        'Amount Received': [10.0, 20.0, 30.0, 5.0],
    })
    result = compute_past_only_history_features_raw(frame)

    # The first two edges share t=10 and both see empty history.
    np.testing.assert_allclose(result[0, :], np.zeros(8), atol=1e-7)
    np.testing.assert_allclose(result[1, :], np.zeros(8), atol=1e-7)

    # At t=20, source 0 has exactly two earlier outgoing edges, while the
    # simultaneous edge from source 3 is not visible.
    assert np.isclose(result[2, 4], np.log1p(2))
    assert np.isclose(result[2, 6], np.log1p(1))
    assert np.isclose(result[3, 5], np.log1p(1))


def test_normalization_uses_train_indices_only_and_preserves_flags():
    frame = pd.DataFrame({
        'from_id': [0, 0, 0, 0],
        'to_id': [1, 2, 1, 2],
        'Timestamp': [10, 20, 30, 40],
        'Amount Received': [10.0, 20.0, 30.0, 1000000.0],
    })
    raw = compute_past_only_history_features_raw(frame)
    normalized, means, stds = normalize_history_features_train_only(raw, [0, 1, 2])

    assert means.shape == (6,)
    assert stds.shape == (6,)
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[:, 2], raw[:, 2])
    np.testing.assert_array_equal(normalized[:, 3], raw[:, 3])


def test_history_ablation_groups_use_canonical_non_overlapping_columns():
    assert resolve_history_feature_indices(['recency']) == (0, 1, 2, 3)
    assert resolve_history_feature_indices(['frequency']) == (4, 5, 6)
    assert resolve_history_feature_indices(['monetary']) == (7,)
    assert resolve_history_feature_indices(
        ['monetary', 'recency', 'frequency']
    ) == tuple(range(8))
    assert history_feature_names(['frequency']) == HISTORY_FEATURE_NAMES[4:7]


def test_endpoint_behavior_preset_excludes_recency_and_pair_frequency():
    assert resolve_history_feature_indices(['endpoint_behavior']) == (4, 5, 7)
    assert history_feature_names(['endpoint_behavior']) == (
        'hist_log_prior_out_count',
        'hist_log_prior_in_count',
        'hist_log_amount_over_prior_out_mean',
    )


def test_endpoint_behavior_preset_cannot_be_mixed_with_ablation_groups():
    try:
        resolve_history_feature_indices(['endpoint_behavior', 'recency'])
    except ValueError:
        pass
    else:
        raise AssertionError('Expected mixed preset/group selection to fail')


def test_unknown_or_ambiguous_history_groups_are_rejected():
    for groups in (['unknown'], ['all', 'recency']):
        try:
            resolve_history_feature_indices(groups)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Expected invalid history groups: {groups}')


def test_reliability_gate_uses_prior_evidence_and_preserves_binary_flags():
    raw = np.zeros((3, len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
    raw[:, 4] = np.log1p([0.0, 5.0, 1000.0])
    raw[:, 5] = np.log1p([0.0, 5.0, 1000.0])
    raw[:, 6] = np.log1p([0.0, 5.0, 1000.0])
    normalized = np.ones_like(raw)
    gated = apply_history_reliability_gate(normalized, raw, kappa=5.0)

    # No history shrinks a continuous statistic to the neutral value.
    assert gated[0, 0] == 0.0
    # Five observations with kappa=5 have reliability 0.5.
    assert np.isclose(gated[1, 0], 0.5, atol=1e-6)
    assert np.isclose(gated[1, 6], 0.5, atol=1e-6)
    # Abundant evidence approaches the original standardized feature.
    assert gated[2, 7] > 0.99
    # Binary availability indicators are not reliability-shrunk.
    np.testing.assert_array_equal(gated[:, 2:4], normalized[:, 2:4])
    # Input arrays are not modified in-place.
    np.testing.assert_array_equal(normalized, np.ones_like(normalized))


def test_reliability_gate_rejects_invalid_kappa_and_shapes():
    valid = np.zeros((1, len(HISTORY_FEATURE_NAMES)), dtype=np.float32)
    for kappa in (0.0, -1.0, np.inf):
        try:
            apply_history_reliability_gate(valid, valid, kappa=kappa)
        except ValueError:
            pass
        else:
            raise AssertionError(f'Expected invalid kappa: {kappa}')

    try:
        apply_history_reliability_gate(np.zeros(8), valid, kappa=5.0)
    except ValueError:
        pass
    else:
        raise AssertionError('Expected invalid normalized feature shape')
