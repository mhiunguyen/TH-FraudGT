"""Build Notebook 16: hypothesis-driven H-causal seed-42 pilot."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / 'notebooks' / 'kaggle' / \
    '15_A_Causal_Prediction_History_Coverage_Audit_Seed42_T4.ipynb'
OUTPUT_NOTEBOOK = ROOT / 'notebooks' / 'kaggle' / \
    '16_H_Causal_EndpointBehavior_Pilot_Seed42_T4.ipynb'


def markdown(text: str) -> dict:
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': [line + '\n' for line in text.strip().splitlines()],
    }


def code(text: str) -> dict:
    return {
        'cell_type': 'code',
        'execution_count': None,
        'metadata': {},
        'outputs': [],
        'source': [line + '\n' for line in text.strip().splitlines()],
    }


source = json.loads(SOURCE_NOTEBOOK.read_text(encoding='utf-8'))
dependency_cell = next(
    cell for cell in source['cells']
    if cell['cell_type'] == 'code'
    and 'Building pyg-lib 0.8.0 CPU sampler backend' in ''.join(cell['source'])
)
dependency_cell = {
    'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [],
    'source': dependency_cell['source'],
}
dependency_text = ''.join(dependency_cell['source']).replace(
    "if RUN_MODE == 'full' and torch.cuda.device_count() < 2:\n"
    "    raise RuntimeError('Full 3-seed yêu cầu Kaggle 2×T4; một GPU có nguy cơ vượt giới hạn phiên.')\n",
    "if torch.cuda.device_count() < 1:\n"
    "    raise RuntimeError('Notebook 16 cần ít nhất một GPU T4.')\n",
)
dependency_cell['source'] = dependency_text.splitlines(keepends=True)

cells = [
    markdown(r'''
# Notebook 16 — H-causal với hành vi quá khứ của tài khoản

Mục tiêu duy nhất: kiểm tra liệu ba đặc trưng lịch sử **strict-past** có khắc phục điểm yếu của `A-causal-uniform` ở nhóm tài khoản hoạt động rất cao hay không.

H-causal chỉ thêm:

1. `log1p(prior_out_count)`: số giao dịch gửi trước đó của tài khoản gửi.
2. `log1p(prior_in_count)`: số giao dịch nhận trước đó của tài khoản nhận.
3. `log1p(current_amount / prior_out_mean)`: độ lệch số tiền hiện tại so với lịch sử gửi.

Không thêm recency vì sampler đã giữ tốt giao dịch gần nhất. Không thêm pair-frequency vì Notebook 15 cho thấy lịch sử đúng cặp gửi–nhận đã được giữ khoảng 97–98%.

Đây là pilot một seed trên **validation**. Notebook không huấn luyện lại A-causal; mốc A cố định lấy từ Notebook 15, commit `880db80`.
'''),
    code(r'''
SEED = 42
RUN_MODE = 'full'
AUDIT_SPLIT = 'val'
MAX_EPOCHS = 200
TRAIN_STEPS = 256
EVAL_PERIOD = 25
BATCH_SIZE = 256
BATCH_ACCUMULATION = 8
WARMUP_EPOCHS = 10
FANOUT = [25, 25]
NUM_THREADS = 2
NUM_WORKERS = 2
HISTORY_GROUPS = ['endpoint_behavior']

assert RUN_MODE == 'full'
assert AUDIT_SPLIT == 'val', 'Không dùng test để quyết định cải tiến.'
assert MAX_EPOCHS * TRAIN_STEPS * BATCH_SIZE == 100 * 256 * 512
assert MAX_EPOCHS * TRAIN_STEPS // BATCH_ACCUMULATION == 100 * 256 // 4
print('H-causal-uniform | seed:', SEED)
print('History preset:', HISTORY_GROUPS)
print('Target exposures:', MAX_EPOCHS * TRAIN_STEPS * BATCH_SIZE)
print('Optimizer updates:', MAX_EPOCHS * TRAIN_STEPS // BATCH_ACCUMULATION)
'''),
    markdown('## 1. Môi trường và dependencies'),
    dependency_cell,
    markdown('## 2. Lấy đúng source, chạy kiểm thử và lấy dữ liệu'),
    code(r'''
from shutil import copy2

REPO_URL = 'https://github.com/mhiunguyen/TH-FraudGT.git'
repo = Path('/kaggle/working/TH-FraudGT')
if not (repo / '.git').exists():
    subprocess.run(['git', 'clone', REPO_URL, str(repo)], check=True)
else:
    subprocess.run(['git', '-C', str(repo), 'pull', '--ff-only'], check=True)
os.chdir(repo)
commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
print('Commit:', commit)

required = [
    repo / 'configs/AML-Small-HI/AML-Small-HI-H16-Causal-EndpointBehavior-T4.yaml',
    repo / 'fraudGT/datasets/history_features.py',
    repo / 'scripts/audit_prediction_history.py',
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise RuntimeError('Source trên Git chưa có thiết kế H-causal: ' + ', '.join(missing))

subprocess.run(
    [sys.executable, '-m', 'pytest', '-q',
     'tests/test_history_features.py',
     'tests/test_temporal_target_batch.py',
     'tests/test_prediction_batch_audit.py',
     'tests/test_prediction_history_audit.py'],
    cwd=repo, check=True)
print('PASS: history features, temporal sampler và audit tests.')

candidates = list(Path('/kaggle/input').rglob('HI-Small_Trans.csv'))
if not candidates:
    raise FileNotFoundError('Hãy Add Input bộ IBM AML; thiếu HI-Small_Trans.csv.')
destination = repo / 'data/AML/HI-Small_Trans.csv'
destination.parent.mkdir(parents=True, exist_ok=True)
if not destination.exists() or destination.stat().st_size != candidates[0].stat().st_size:
    copy2(candidates[0], destination)
print('Dataset:', destination, '| MiB:', round(destination.stat().st_size / 1024**2, 1))
'''),
    markdown('## 3. Khóa cấu hình H-causal và kiểm tra tính công bằng'),
    code(r'''
import yaml

tracked_cfg = repo / 'configs/AML-Small-HI/AML-Small-HI-H16-Causal-EndpointBehavior-T4.yaml'
cfg_dict = yaml.safe_load(tracked_cfg.read_text(encoding='utf-8'))
cfg_dict['out_dir'] = str(repo / 'results_nb16_h_causal')
cfg_dict['dataset']['dir'] = str(repo / 'data')
cfg_dict['seed'] = SEED
cfg_dict['num_threads'] = NUM_THREADS
cfg_dict['num_workers'] = NUM_WORKERS
cfg_dict['wandb']['use'] = False

baseline_path = repo / 'configs/AML-Small-HI/AML-Small-HI-A14R-causal-uniform-clean.yaml'
baseline = yaml.safe_load(baseline_path.read_text(encoding='utf-8'))

same_paths = [
    ('seed',),
    ('train', 'sampler'), ('train', 'temporal_strategy'),
    ('train', 'temporal_strict'), ('train', 'neighbor_sizes'),
    ('train', 'iter_per_epoch'), ('train', 'batch_size'),
    ('train', 'eval_period'), ('val', 'sampler'), ('val', 'iter_per_epoch'),
    ('model', 'loss_fun'), ('model', 'loss_fun_weight'),
    ('gt', 'layers'), ('gt', 'attn_heads'), ('gt', 'dim_hidden'),
    ('optim', 'base_lr'), ('optim', 'max_epoch'),
    ('optim', 'batch_accumulation'), ('optim', 'num_warmup_epochs'),
]
def nested_get(mapping, path):
    for key in path:
        mapping = mapping[key]
    return mapping

for path in same_paths:
    assert nested_get(cfg_dict, path) == nested_get(baseline, path), path
assert baseline['dataset']['add_history'] is False
assert cfg_dict['dataset']['add_history'] is True
assert cfg_dict['dataset']['history_groups'] == HISTORY_GROUPS
assert cfg_dict['dataset']['history_reliability'] is False

generated = Path('/kaggle/working/generated_configs_nb16')
generated.mkdir(exist_ok=True)
CFG = generated / 'AML-Small-HI-H16-causal-endpoint-behavior.yaml'
CFG.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding='utf-8')
print('PASS: A và H chỉ khác phần bổ sung endpoint history.')
print('Config:', CFG)
'''),
    markdown('## 4. Tạo history cache strict-past và kiểm tra ba đặc trưng'),
    code(r'''
import gc, math

processed = repo / 'data/AML/Small-HI/processed'
for filename in ['data_history.pt', 'ports_history.pt']:
    path = processed / filename
    if path.exists():
        path.unlink()
        print('Removed stale history cache:', path)

sys.path.insert(0, str(repo))
from fraudGT.datasets.aml_dataset import AMLDataset

started = time.time()
dataset = AMLDataset(
    root=str(repo / 'data/AML'), name='Small-HI',
    reverse_mp=True, add_ports=True, add_history=True,
    history_groups=HISTORY_GROUPS, history_reliability=False)
expected_names = [
    'hist_log_prior_out_count',
    'hist_log_prior_in_count',
    'hist_log_amount_over_prior_out_mean',
]
assert dataset.history_feature_names == expected_names
task = ('node', 'to', 'node')
counts = {
    split: int(dataset[split][task].split_mask.sum())
    for split in ['train', 'val', 'test']
}
batches = {split: math.ceil(count / BATCH_SIZE) for split, count in counts.items()}
print('Selected history fields:', dataset.history_feature_names)
print('Target edges:', counts)
print('Full-split batches:', batches)
print(f'History cache ready in {(time.time()-started)/60:.1f} min')
del dataset
gc.collect()
'''),
    markdown(r'''
## 5. Huấn luyện H-causal seed 42

Heartbeat `epoch x/200` là tiến độ train. Dòng GPU utilization chỉ là mức GPU đang bận, không phải phần trăm hoàn thành.
'''),
    code(r'''
import re

LOG = Path('/kaggle/working/NB16_h_causal_seed42.log')
TAG = 'NB16-full'
command = [
    sys.executable, '-u', '-m', 'fraudGT.main',
    '--cfg', str(CFG), '--repeat', '1', '--gpu', '0',
    'name_tag', TAG,
]

def tail_text(path, max_bytes=512_000):
    if not path.exists():
        return ''
    with path.open('rb') as stream:
        stream.seek(0, 2)
        size = stream.tell()
        stream.seek(max(0, size - max_bytes))
        return stream.read().decode('utf-8', errors='replace')

with LOG.open('w', encoding='utf-8') as stream:
    process = subprocess.Popen(
        command, cwd=repo, stdout=stream, stderr=subprocess.STDOUT, text=True)
print('Started H-causal seed 42 | PID', process.pid)
started = time.time()
while process.poll() is None:
    time.sleep(60)
    recent = tail_text(LOG)
    epochs = re.findall(r"train:\s*\{'epoch':\s*(\d+)", recent)
    if epochs:
        completed = int(epochs[-1]) + 1
        progress = f'{completed}/{MAX_EPOCHS} epochs ({100*completed/MAX_EPOCHS:.1f}%)'
    else:
        progress = 'đang khởi tạo/cache'
    print(f'[heartbeat] {(time.time()-started)/60:.0f} min | train progress: {progress}', flush=True)
    subprocess.run([
        'nvidia-smi', '--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader'], check=False)

if process.returncode != 0:
    print('\n'.join(tail_text(LOG).splitlines()[-120:]))
    raise RuntimeError(f'Training failed with code {process.returncode}')
log_lower = LOG.read_text(encoding='utf-8', errors='replace').lower()
if "loss': nan" in log_lower or 'contains nan/inf' in log_lower or 'no supervision' in log_lower:
    raise RuntimeError('Numerical/supervision audit failed.')
if 'task done' not in log_lower or '[*] all done' not in log_lower:
    raise RuntimeError('Training ended without completion markers.')
print('PASS: H-causal seed 42 training completed.')
'''),
    markdown('## 6. Chọn epoch và threshold bằng validation, rồi sao lưu checkpoint'),
    code(r'''
import pandas as pd

RUN_ROOT = repo / 'results_nb16_h_causal' / f'{CFG.stem}-{TAG}-gpu0'
RUN_DIR = RUN_ROOT / str(SEED)
SUMMARY = Path('/kaggle/working/summary_NB16_h_causal_validation_selected_seed42.csv')
subprocess.run([
    sys.executable, str(repo / 'scripts/summarize_thresholds.py'),
    str(RUN_ROOT), '--output', str(SUMMARY)], check=True)
selection = pd.read_csv(SUMMARY)
assert selection['seed'].astype(int).tolist() == [SEED]
best_epoch = int(selection.loc[0, 'best_epoch'])
threshold = float(selection.loc[0, 'threshold'])
checkpoint = RUN_DIR / 'ckpt' / f'{best_epoch}.ckpt'
assert checkpoint.exists(), f'Missing selected checkpoint: {checkpoint}'
SAFE_CHECKPOINT = Path('/kaggle/working/NB16_H_causal_selected_seed42.ckpt')
SAFE_CONFIG = Path('/kaggle/working/AML-Small-HI-H16-causal-endpoint-behavior.yaml')
copy2(checkpoint, SAFE_CHECKPOINT)
copy2(CFG, SAFE_CONFIG)
assert SAFE_CHECKPOINT.exists() and SAFE_CHECKPOINT.stat().st_size > 0
display(selection[['seed', 'best_epoch', 'threshold', 'val_f1']])
print('Checkpoint backup saved before audit:', SAFE_CHECKPOINT)
'''),
    markdown('## 7. Audit H-causal trên toàn bộ validation'),
    code(r'''
AUDIT_DIR = Path('/kaggle/working/NB16_h_causal_prediction_history_audit')
AUDIT_DIR.mkdir(exist_ok=True)
AUDIT_LOG = Path('/kaggle/working/NB16_h_causal_prediction_history_audit.log')
audit_command = [
    sys.executable, '-u', str(repo / 'scripts/audit_prediction_history.py'),
    '--cfg', str(CFG),
    '--run-dir', str(RUN_DIR),
    '--selection-summary', str(SUMMARY),
    '--output-dir', str(AUDIT_DIR),
    '--split', AUDIT_SPLIT,
    '--seed', str(SEED),
    '--device', 'cuda:0',
]
with AUDIT_LOG.open('w', encoding='utf-8') as stream:
    completed = subprocess.run(
        audit_command, cwd=repo, stdout=stream,
        stderr=subprocess.STDOUT, text=True)
if completed.returncode != 0:
    print('\n'.join(AUDIT_LOG.read_text(errors='replace').splitlines()[-150:]))
    raise RuntimeError(f'Prediction audit failed with code {completed.returncode}')
print('\n'.join(AUDIT_LOG.read_text(errors='replace').splitlines()[-80:]))
print('PASS: H-causal validation audit completed.')
'''),
    markdown('## 8. So sánh với A-causal cố định từ Notebook 15'),
    code(r'''
from IPython.display import Image, display
import json
import matplotlib.pyplot as plt

DETAIL = AUDIT_DIR / f'prediction_history_audit_{AUDIT_SPLIT}_seed{SEED}.csv'
BY_HISTORY = AUDIT_DIR / f'prediction_by_history_group_{AUDIT_SPLIT}_seed{SEED}.csv'
BY_COVERAGE = AUDIT_DIR / f'prediction_by_coverage_group_{AUDIT_SPLIT}_seed{SEED}.csv'
GLOBAL = AUDIT_DIR / f'prediction_audit_global_{AUDIT_SPLIT}_seed{SEED}.json'
FIGURE = AUDIT_DIR / f'prediction_history_audit_{AUDIT_SPLIT}_seed{SEED}.png'

h_global = json.loads(GLOBAL.read_text())
h_history = pd.read_csv(BY_HISTORY)

# Frozen reference from completed Notebook 15, commit 880db80.
a_global = {
    'model': 'A-causal', 'precision': 0.5226244343891403,
    'recall': 0.44594594594594594, 'f1': 0.48125,
    'ap': 0.4233974498682437, 'auc': 0.9855870790660803,
}
a_high = pd.DataFrame([
    {'history_group': '1,001–10,000', 'model': 'A-causal', 'n': 15510,
     'fraud': 27, 'recall': 0.0, 'f1': 0.0, 'ap': 0.001728},
    {'history_group': '>10,000', 'model': 'A-causal', 'n': 83942,
     'fraud': 106, 'recall': 0.0, 'f1': 0.0, 'ap': 0.001163},
])
h_global_row = {'model': 'H-causal', **{
    key: h_global[key] for key in ['precision', 'recall', 'f1', 'ap', 'auc']}}
GLOBAL_COMPARE = Path('/kaggle/working/comparison_NB16_A_vs_H_global_val_seed42.csv')
global_compare = pd.DataFrame([a_global, h_global_row])
global_compare.to_csv(GLOBAL_COMPARE, index=False)

high_groups = ['1,001–10,000', '>10,000']
h_high = h_history[h_history['history_group'].isin(high_groups)].copy()
h_high.insert(1, 'model', 'H-causal')
HISTORY_COMPARE = Path('/kaggle/working/comparison_NB16_A_vs_H_high_history_val_seed42.csv')
history_compare = pd.concat([
    a_high, h_high[a_high.columns]], ignore_index=True)
history_compare.to_csv(HISTORY_COMPARE, index=False)

print('Toàn validation:')
display(global_compare)
print('Hai nhóm lịch sử lớn:')
display(history_compare)

COMPARISON_FIGURE = Path('/kaggle/working/comparison_NB16_A_vs_H_val_seed42.png')
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
global_compare.set_index('model')[['f1', 'recall', 'ap']].plot(
    kind='bar', rot=0, ax=axes[0], color=['#4472C4', '#ED7D31', '#70AD47'])
axes[0].set_title('Toàn validation'); axes[0].set_ylabel('Score')
for metric, axis, title in [('recall', axes[1], 'Recall nhóm lịch sử lớn'),
                            ('ap', axes[2], 'AP nhóm lịch sử lớn')]:
    history_compare.pivot(index='history_group', columns='model', values=metric).plot(
        kind='bar', rot=15, ax=axis, color=['#4472C4', '#ED7D31'])
    axis.set_title(title); axis.set_xlabel('Số giao dịch quá khứ')
for axis in axes:
    axis.grid(axis='y', alpha=.25)
fig.suptitle('Notebook 16 — A-causal và H-causal, validation seed 42')
fig.tight_layout()
fig.savefig(COMPARISON_FIGURE, dpi=180, bbox_inches='tight')
plt.close(fig)
display(Image(filename=str(COMPARISON_FIGURE)))

delta_f1 = h_global['f1'] - a_global['f1']
delta_ap = h_global['ap'] - a_global['ap']
print(f'ΔF1 H−A = {delta_f1:+.5f} | ΔAP H−A = {delta_ap:+.5f}')
print('Chỉ tiếp tục 3 seed nếu H cải thiện AP/F1 toàn validation và/hoặc phục hồi rõ nhóm >1,000.')
print('Một seed chỉ dùng khóa thiết kế, chưa phải kết luận cuối.')
'''),
    markdown('## 9. Đóng gói artifact'),
    code(r'''
import shutil

evidence = [
    CFG, SAFE_CONFIG, SAFE_CHECKPOINT, LOG, AUDIT_LOG, SUMMARY,
    DETAIL, BY_HISTORY, BY_COVERAGE, GLOBAL, FIGURE,
    GLOBAL_COMPARE, HISTORY_COMPARE, COMPARISON_FIGURE,
]
bundle = Path('/kaggle/working/NB16_H_causal_endpoint_behavior_seed42_artifacts')
bundle.mkdir(exist_ok=True)
for path in evidence:
    if path.exists():
        shutil.copy2(path, bundle / path.name)
(bundle / 'commit.txt').write_text(commit + '\n', encoding='utf-8')
archive = shutil.make_archive(str(bundle), 'zip', bundle)
print('Download:', archive)
'''),
    markdown(r'''
## Sau Notebook 16

- H-causal cải thiện validation và đặc biệt nhóm lịch sử lớn: khóa kiến trúc này, chạy A-causal/H-causal trên seeds 42–44 rồi mới báo test.
- H-causal không cải thiện: bác bỏ ba đặc trưng này; không tiếp tục tinh chỉnh nhiều biến thể trên test.
'''),
]

metadata = json.loads(json.dumps(source.get('metadata', {})))
metadata['title'] = 'Notebook 16 - H-causal endpoint behavior pilot, seed 42'
metadata.setdefault('kaggle', {})['accelerator'] = 'gpu'
metadata['kaggle']['isGpuEnabled'] = True
metadata['kaggle']['isInternetEnabled'] = True

notebook = {
    'cells': cells,
    'metadata': metadata,
    'nbformat': 4,
    'nbformat_minor': 5,
}
OUTPUT_NOTEBOOK.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(OUTPUT_NOTEBOOK)
