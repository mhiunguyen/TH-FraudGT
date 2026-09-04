"""Past-only historical features for timestamped AML transactions.

All statistics for an edge at time ``t`` are computed from edges with a
strictly smaller timestamp. Edges that share the same timestamp are therefore
not allowed to observe one another.
"""

from __future__ import annotations

from typing import Iterable, Sequence, Tuple

import numpy as np
import pandas as pd


HISTORY_FEATURE_NAMES = (
    "hist_log_seconds_since_prev_out",
    "hist_log_seconds_since_prev_in",
    "hist_has_prev_out",
    "hist_has_prev_in",
    "hist_log_prior_out_count",
    "hist_log_prior_in_count",
    "hist_log_prior_pair_count",
    "hist_log_amount_over_prior_out_mean",
)

# The ablation groups are defined once here so every experiment uses the
# exact same column mapping.  Columns remain in the canonical eight-feature
# order regardless of the order used in a YAML file.
HISTORY_FEATURE_GROUPS = {
    "recency": (0, 1, 2, 3),
    "frequency": (4, 5, 6),
    "monetary": (7,),
}

# A hypothesis-driven preset for the causal history experiment.  Notebook 15
# found that high-activity endpoint history is heavily truncated while direct
# pair history is almost completely retained.  This preset therefore exposes
# only endpoint activity and amount deviation, excluding recency and pair
# frequency rather than adding all eight historical features.
HISTORY_FEATURE_PRESETS = {
    "endpoint_behavior": (4, 5, 7),
}

# Binary indicators stay in {0, 1}; the other columns are standardized.
CONTINUOUS_HISTORY_COLUMNS = (0, 1, 4, 5, 6, 7)

# Each continuous feature is shrunk by the amount of evidence available for
# the entity/pair that produced it.  The count columns contain log1p(count) in
# the raw feature matrix, so no additional pass over the transaction table is
# needed.
_RELIABILITY_COUNT_COLUMN = {
    0: 4,  # outgoing recency <- prior outgoing count
    1: 5,  # incoming recency <- prior incoming count
    4: 4,  # outgoing frequency
    5: 5,  # incoming frequency
    6: 6,  # pair frequency
    7: 4,  # amount ratio <- prior outgoing history
}


def resolve_history_feature_indices(
    groups: Sequence[str] | str | None = None,
) -> Tuple[int, ...]:
    """Resolve R/F/M group names to canonical history-feature columns.

    ``None``, an empty sequence, or ``"all"`` selects all eight features.
    The function rejects unknown group names instead of silently running an
    invalid ablation.
    """
    if groups is None:
        return tuple(range(len(HISTORY_FEATURE_NAMES)))
    if isinstance(groups, str):
        groups = [part.strip() for part in groups.split(",") if part.strip()]
    normalized = {str(group).strip().lower() for group in groups}
    if not normalized or normalized == {"all"}:
        return tuple(range(len(HISTORY_FEATURE_NAMES)))
    if "all" in normalized:
        raise ValueError("history group 'all' cannot be combined with other groups")
    preset_names = normalized.intersection(HISTORY_FEATURE_PRESETS)
    if preset_names:
        if len(normalized) != 1:
            raise ValueError(
                "history feature presets cannot be combined with other groups"
            )
        return HISTORY_FEATURE_PRESETS[next(iter(preset_names))]
    unknown = normalized.difference(HISTORY_FEATURE_GROUPS)
    if unknown:
        valid = ", ".join((*HISTORY_FEATURE_GROUPS, *HISTORY_FEATURE_PRESETS))
        raise ValueError(
            f"Unknown history feature groups: {sorted(unknown)}; valid: {valid}"
        )
    return tuple(
        column
        for group, columns in HISTORY_FEATURE_GROUPS.items()
        if group in normalized
        for column in columns
    )


def history_feature_names(
    groups: Sequence[str] | str | None = None,
) -> Tuple[str, ...]:
    """Return selected feature names in their canonical column order."""
    return tuple(
        HISTORY_FEATURE_NAMES[column]
        for column in resolve_history_feature_indices(groups)
    )


def _previous_distinct_timestamp(
    frame: pd.DataFrame, key_columns: list[str]
) -> np.ndarray:
    """Return the previous strictly earlier timestamp for each row."""
    keys = key_columns + ["Timestamp"]
    unique_times = frame.loc[:, keys].drop_duplicates()
    unique_times = unique_times.sort_values(keys, kind="stable")
    unique_times["_prev_timestamp"] = unique_times.groupby(
        key_columns, sort=False
    )["Timestamp"].shift(1)
    row_order = frame.loc[:, keys].copy()
    row_order["_row_order"] = np.arange(len(frame), dtype=np.int64)
    merged = row_order.merge(unique_times, on=keys, how="left", sort=False)
    merged = merged.sort_values("_row_order", kind="stable")
    return merged["_prev_timestamp"].to_numpy(dtype=np.float64)


def compute_past_only_history_features_raw(df_edges: pd.DataFrame) -> np.ndarray:
    """Compute eight leakage-safe history features in original row order.

    Required columns are ``from_id``, ``to_id``, ``Timestamp`` and
    ``Amount Received``. The implementation is vectorized for the multi-million
    edge AML files used by the project.
    """
    required = {"from_id", "to_id", "Timestamp", "Amount Received"}
    missing = required.difference(df_edges.columns)
    if missing:
        raise ValueError(f"Missing columns for history features: {sorted(missing)}")

    frame = df_edges.loc[:, sorted(required)].copy()
    frame["_row_order"] = np.arange(len(frame), dtype=np.int64)
    frame = frame.sort_values(
        ["Timestamp", "_row_order"], kind="stable"
    ).reset_index(drop=True)

    # Counts before the whole (entity, timestamp) group exclude every edge at
    # the current timestamp, not just the current row.
    out_position = frame.groupby("from_id", sort=False).cumcount()
    out_same_time_position = frame.groupby(
        ["from_id", "Timestamp"], sort=False
    ).cumcount()
    prior_out_count = (out_position - out_same_time_position).to_numpy(np.float64)

    in_position = frame.groupby("to_id", sort=False).cumcount()
    in_same_time_position = frame.groupby(
        ["to_id", "Timestamp"], sort=False
    ).cumcount()
    prior_in_count = (in_position - in_same_time_position).to_numpy(np.float64)

    pair_position = frame.groupby(["from_id", "to_id"], sort=False).cumcount()
    pair_same_time_position = frame.groupby(
        ["from_id", "to_id", "Timestamp"], sort=False
    ).cumcount()
    prior_pair_count = (pair_position - pair_same_time_position).to_numpy(
        np.float64
    )

    amounts = frame["Amount Received"].astype(np.float64)
    out_cumulative_amount = amounts.groupby(frame["from_id"], sort=False).cumsum()
    out_same_time_amount = amounts.groupby(
        [frame["from_id"], frame["Timestamp"]], sort=False
    ).cumsum()
    prior_out_amount = (out_cumulative_amount - out_same_time_amount).to_numpy()

    prev_out_time = _previous_distinct_timestamp(frame, ["from_id"])
    prev_in_time = _previous_distinct_timestamp(frame, ["to_id"])
    timestamps = frame["Timestamp"].to_numpy(dtype=np.float64)
    has_prev_out = ~np.isnan(prev_out_time)
    has_prev_in = ~np.isnan(prev_in_time)
    out_gap = np.where(has_prev_out, timestamps - prev_out_time, 0.0)
    in_gap = np.where(has_prev_in, timestamps - prev_in_time, 0.0)

    prior_out_mean = np.divide(
        prior_out_amount,
        prior_out_count,
        out=np.zeros_like(prior_out_amount),
        where=prior_out_count > 0,
    )
    amount_ratio = np.divide(
        amounts.to_numpy(),
        prior_out_mean,
        out=np.zeros(len(frame), dtype=np.float64),
        where=prior_out_mean > 0,
    )
    amount_ratio = np.clip(amount_ratio, 0.0, 1e6)

    features = np.column_stack(
        (
            np.log1p(np.maximum(out_gap, 0.0)),
            np.log1p(np.maximum(in_gap, 0.0)),
            has_prev_out.astype(np.float64),
            has_prev_in.astype(np.float64),
            np.log1p(prior_out_count),
            np.log1p(prior_in_count),
            np.log1p(prior_pair_count),
            np.log1p(amount_ratio),
        )
    )
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    # Restore the exact order supplied by the caller.
    inverse_order = np.argsort(frame["_row_order"].to_numpy(), kind="stable")
    return features[inverse_order].astype(np.float32, copy=False)


def normalize_history_features_train_only(
    features: np.ndarray, train_indices: Iterable[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize continuous history columns using training edges only."""
    features = np.asarray(features, dtype=np.float32)
    if hasattr(train_indices, "detach"):
        train_indices = train_indices.detach().cpu().numpy()
    train_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    if features.ndim != 2 or features.shape[1] != len(HISTORY_FEATURE_NAMES):
        raise ValueError(
            f"Expected history feature shape [N, {len(HISTORY_FEATURE_NAMES)}], "
            f"got {features.shape}"
        )
    if train_indices.size == 0:
        raise ValueError("train_indices must not be empty")

    continuous = np.asarray(CONTINUOUS_HISTORY_COLUMNS, dtype=np.int64)
    train_values = features[train_indices[:, None], continuous]
    means = train_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    stds = train_values.std(axis=0, dtype=np.float64).astype(np.float32)
    stds = np.where(stds > 0, stds, np.float32(1.0))

    normalized = features.copy()
    normalized[:, continuous] = (
        normalized[:, continuous] - means[None, :]
    ) / stds[None, :]
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized.astype(np.float32, copy=False), means, stds


def apply_history_reliability_gate(
    normalized_features: np.ndarray,
    raw_features: np.ndarray,
    kappa: float = 5.0,
) -> np.ndarray:
    """Shrink uncertain historical features towards their neutral value.

    The reliability of a statistic backed by ``n`` prior observations is
    ``q = n / (n + kappa)``.  The gate is applied after train-only
    standardization, so zero is the neutral (training-mean) value.  Binary
    ``has previous`` indicators are intentionally left unchanged.

    Args:
        normalized_features: The eight features after train-only scaling.
        raw_features: The same eight features before scaling. Count columns
            must still contain ``log1p(prior_count)``.
        kappa: Positive shrinkage strength. Larger values require more past
            observations before trusting a historical statistic.
    """
    if not np.isfinite(kappa) or float(kappa) <= 0:
        raise ValueError(f"kappa must be finite and > 0, got {kappa}")

    normalized = np.asarray(normalized_features, dtype=np.float32)
    raw = np.asarray(raw_features, dtype=np.float32)
    if normalized.ndim != 2 or normalized.shape[1] != len(HISTORY_FEATURE_NAMES):
        raise ValueError(
            f"Expected normalized feature shape [N, {len(HISTORY_FEATURE_NAMES)}], "
            f"got {normalized.shape}"
        )
    expected_shape = (normalized.shape[0], len(HISTORY_FEATURE_NAMES))
    if raw.shape != expected_shape:
        raise ValueError(
            f"raw_features must have shape {expected_shape}, got {raw.shape}"
        )

    gated = normalized.copy()
    for feature_column, count_column in _RELIABILITY_COUNT_COLUMN.items():
        prior_count = np.expm1(raw[:, count_column].astype(np.float64))
        prior_count = np.maximum(prior_count, 0.0)
        reliability = prior_count / (prior_count + float(kappa))
        gated[:, feature_column] *= reliability.astype(np.float32)

    return np.nan_to_num(
        gated, nan=0.0, posinf=0.0, neginf=0.0
    ).astype(np.float32, copy=False)
