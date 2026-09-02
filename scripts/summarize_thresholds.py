"""Choose threshold/epoch on validation logs and report matching test metrics."""

import argparse
import json
import math
import re
from pathlib import Path

import pandas as pd


def read_json_lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=Path,
                        help='Directory containing one subdirectory per seed')
    parser.add_argument('--output', type=Path, default=Path('summary_mvia.csv'))
    parser.add_argument(
        '--fixed-threshold', type=float,
        help='Use one threshold for every model. This can summarize legacy '
             'logs at the config threshold 0.10 without retraining.')
    args = parser.parse_args()

    rows = []
    # Ignore GraphGym's aggregate directory. Only numeric directories are
    # individual runs (for example 42, 43 and 44).
    seed_dirs = sorted(
        (path for path in args.run_dir.iterdir()
         if path.is_dir() and path.name.isdigit()),
        key=lambda path: int(path.name),
    )
    for seed_dir in seed_dirs:
        val_path = seed_dir / 'val' / 'stats.json'
        test_path = seed_dir / 'test' / 'stats.json'
        if not val_path.exists() or not test_path.exists():
            continue

        val_records = read_json_lines(val_path)
        test_records = read_json_lines(test_path)
        invalid = [
            (split, record.get('epoch'))
            for split, records in [('val', val_records),
                                   ('test', test_records)]
            for record in records
            if not math.isfinite(float(record.get('loss', float('nan'))))
        ]
        if invalid:
            raise RuntimeError(
                f'Invalid NaN/Inf evaluation records in {seed_dir}: '
                f'{invalid}. Refusing to summarize a failed run.')
        candidates = []
        if args.fixed_threshold is not None:
            threshold_percent = int(round(args.fixed_threshold * 100))
            threshold_key = f'f1_t{threshold_percent:02d}'
            for record in val_records:
                # Old logs only contain f1 at cfg.model.thresh=0.10.
                if threshold_key in record:
                    value = record[threshold_key]
                elif threshold_percent == 10:
                    value = record['f1']
                else:
                    raise RuntimeError(
                        f'{threshold_key} is absent from legacy log {val_path}')
                candidates.append((float(value), threshold_percent, record))
        else:
            for record in val_records:
                for key, value in record.items():
                    match = re.fullmatch(r'f1_t(\d+)', key)
                    if match:
                        candidates.append(
                            (float(value), int(match.group(1)), record))
            if not candidates:
                raise RuntimeError(
                    f'No threshold metrics found in {val_path}. For an old '
                    'run made with model.thresh=0.10, add '
                    '--fixed-threshold 0.10.')

        # Select both epoch and threshold using validation F1 only.
        val_f1, threshold_percent, val_record = max(
            candidates, key=lambda item: (item[0], item[1]))
        epoch = val_record['epoch']
        test_record = next(
            (record for record in test_records
             if record['epoch'] == epoch), None)
        if test_record is None:
            raise RuntimeError(f'No test record for epoch {epoch} in {test_path}')

        suffix = f'{threshold_percent:02d}'
        metric_suffix = f'_t{suffix}' if f'f1_t{suffix}' in test_record else ''
        rows.append({
            'seed': seed_dir.name,
            'best_epoch': epoch,
            'threshold': threshold_percent / 100,
            'val_f1': val_f1,
            'test_f1': test_record[f'f1{metric_suffix}'],
            'test_precision': test_record[f'precision{metric_suffix}'],
            'test_recall': test_record[f'recall{metric_suffix}'],
            'test_auc': test_record['auc'],
            'test_ap': test_record.get('ap'),
            'test_accuracy_default_threshold': test_record['accuracy'],
        })

    if not rows:
        raise RuntimeError(f'No completed seed directories found in {args.run_dir}')

    frame = pd.DataFrame(rows)
    frame.to_csv(args.output, index=False)
    print(frame.to_string(index=False))
    print('\nMean ± sample standard deviation')
    for metric in ('test_f1', 'test_precision', 'test_recall', 'test_auc',
                   'test_ap'):
        print(f'{metric}: {frame[metric].mean():.4f} ± {frame[metric].std(ddof=1):.4f}')
    print(f'\nCSV: {args.output.resolve()}')


if __name__ == '__main__':
    main()
