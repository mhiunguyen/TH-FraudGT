"""Build the self-contained Kaggle Notebook 15 artifact."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_NOTEBOOK = ROOT / 'notebooks' / 'kaggle' / \
    '14R_A_Noncausal_Uniform_vs_Causal_Uniform_ComputeMatched_3Seeds_T4x2.ipynb'
OUTPUT_NOTEBOOK = ROOT / 'notebooks' / 'kaggle' / \
    '15_A_Causal_Prediction_History_Coverage_Audit_Seed42_T4.ipynb'


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

cells = [
    markdown(r'''
# Notebook 15 — Prediction error theo history và sampling coverage

Mục tiêu duy nhất: kiểm tra trên **validation** xem các giao dịch có lịch sử lớn nhưng sampler chỉ quan sát được ít lịch sử có bị dự đoán kém hơn không.

Pipeline:

1. Huấn luyện lại đúng `A-causal-uniform-clean`, seed 42 và ngân sách compute-matched của Notebook 14R.
2. Chọn epoch và threshold hoàn toàn bằng validation.
3. Nạp checkpoint đã chọn, chạy inference toàn bộ validation với `shuffle=False`.
4. Với từng giao dịch, ghép prediction với lịch sử strict-past thực và phần lịch sử sampler quan sát được.
5. Báo cáo theo nhóm history/coverage.

Notebook **không dùng test để quyết định cải tiến**. Nếu validation ủng hộ giả thuyết history, bước sau mới khóa thiết kế H-causal và chạy thí nghiệm xác nhận cuối trên test.
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

assert RUN_MODE == 'full'
assert AUDIT_SPLIT == 'val', 'Không dùng test để lựa chọn hướng cải tiến.'
assert MAX_EPOCHS * TRAIN_STEPS * BATCH_SIZE == 100 * 256 * 512
assert MAX_EPOCHS * TRAIN_STEPS // BATCH_ACCUMULATION == 100 * 256 // 4
print('A-causal-uniform | seed:', SEED)
print('Audit split:', AUDIT_SPLIT)
print('Effective batch:', BATCH_SIZE * BATCH_ACCUMULATION)
print('Target exposures:', MAX_EPOCHS * TRAIN_STEPS * BATCH_SIZE)
print('Optimizer updates:', MAX_EPOCHS * TRAIN_STEPS // BATCH_ACCUMULATION)
'''),
    markdown('## 1. Môi trường và dependencies'),
    dependency_cell,
    markdown('## 2. Lấy đúng source và dữ liệu AML-Small-HI'),
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
    repo / 'scripts/audit_prediction_history.py',
    repo / 'scripts/history_count_utils.py',
    repo / 'fraudGT/sampler/custom_sampler.py',
]
missing = [str(path) for path in required if not path.exists()]
if missing:
    raise RuntimeError('Source trên Git chưa có Notebook 15: ' + ', '.join(missing))

subprocess.run(
    [sys.executable, '-m', 'pytest', '-q',
     'tests/test_temporal_target_batch.py',
     'tests/test_prediction_batch_audit.py',
     'tests/test_prediction_history_audit.py'],
    cwd=repo, check=True)
print('PASS: temporal sampler và strict-history counting tests.')

candidates = list(Path('/kaggle/input').rglob('HI-Small_Trans.csv'))
if not candidates:
    raise FileNotFoundError('Hãy Add Input bộ IBM AML; thiếu HI-Small_Trans.csv.')
destination = repo / 'data/AML/HI-Small_Trans.csv'
destination.parent.mkdir(parents=True, exist_ok=True)
if not destination.exists() or destination.stat().st_size != candidates[0].stat().st_size:
    copy2(candidates[0], destination)
print('Dataset:', destination, '| MiB:', round(destination.stat().st_size / 1024**2, 1))
'''),
    markdown('## 3. Sinh cấu hình A-causal-uniform seed 42'),
    code(r'''
import copy, yaml

base_path = repo / 'configs/AML-Small-HI/AML-Small-HI-A-Retrain-T4.yaml'
cfg_dict = yaml.safe_load(base_path.read_text(encoding='utf-8'))
cfg_dict['out_dir'] = str(repo / 'results_nb15_history_audit')
cfg_dict['seed'] = SEED
cfg_dict['dataset']['dir'] = str(repo / 'data')
cfg_dict['dataset']['add_history'] = False
cfg_dict['num_threads'] = NUM_THREADS
cfg_dict['num_workers'] = NUM_WORKERS
cfg_dict['wandb']['use'] = False
cfg_dict['train']['sampler'] = 'temporal_link_neighbor'
cfg_dict['val']['sampler'] = 'temporal_link_neighbor'
cfg_dict['train']['temporal_strategy'] = 'uniform'
cfg_dict['train']['temporal_strict'] = True
cfg_dict['train']['neighbor_sizes'] = FANOUT
cfg_dict['train']['batch_size'] = BATCH_SIZE
cfg_dict['train']['iter_per_epoch'] = TRAIN_STEPS
cfg_dict['train']['eval_period'] = EVAL_PERIOD
cfg_dict['train']['persistent_workers'] = False
cfg_dict['train']['enable_ckpt'] = True
cfg_dict['train']['ckpt_best'] = True
cfg_dict['train']['auto_resume'] = True
cfg_dict['train']['ckpt_resume_period'] = EVAL_PERIOD
cfg_dict['train']['ckpt_clean'] = False  # giữ checkpoint mỗi lần eval để nạp đúng epoch validation chọn
cfg_dict['val']['iter_per_epoch'] = -1
cfg_dict['optim']['max_epoch'] = MAX_EPOCHS
cfg_dict['optim']['batch_accumulation'] = BATCH_ACCUMULATION
cfg_dict['optim']['num_warmup_epochs'] = WARMUP_EPOCHS
cfg_dict['mvia']['thresholds'] = [round(x / 100, 2) for x in range(5, 100, 5)]

generated = Path('/kaggle/working/generated_configs_nb15')
generated.mkdir(exist_ok=True)
CFG = generated / 'AML-Small-HI-A15-causal-uniform-history-audit.yaml'
CFG.write_text(yaml.safe_dump(cfg_dict, sort_keys=False), encoding='utf-8')
loaded = yaml.safe_load(CFG.read_text())
assert loaded['train']['sampler'] == 'temporal_link_neighbor'
assert loaded['train']['temporal_strategy'] == 'uniform'
assert loaded['train']['temporal_strict'] is True
assert loaded['val']['iter_per_epoch'] == -1
assert loaded['train']['ckpt_clean'] is False
print('Config:', CFG)
print('Sampler: causal uniform, strict-past | epochs:', MAX_EPOCHS)
'''),
    markdown('## 4. Tạo clean cache và kiểm tra split'),
    code(r'''
import gc, math

processed = repo / 'data/AML/Small-HI/processed'
for filename in ['data.pt', 'ports.pt']:
    path = processed / filename
    if path.exists():
        path.unlink()
        print('Removed stale cache:', path)

sys.path.insert(0, str(repo))
from fraudGT.datasets.aml_dataset import AMLDataset

started = time.time()
dataset = AMLDataset(
    root=str(repo / 'data/AML'), name='Small-HI',
    reverse_mp=True, add_ports=True, add_history=False)
task = ('node', 'to', 'node')
counts = {
    split: int(dataset[split][task].split_mask.sum())
    for split in ['train', 'val', 'test']
}
batches = {split: math.ceil(count / BATCH_SIZE) for split, count in counts.items()}
print('Target edges:', counts)
print('Full-split batches:', batches)
print(f'Clean cache ready in {(time.time()-started)/60:.1f} min')
del dataset
gc.collect()
'''),
    markdown(r'''
## 5. Huấn luyện một seed

Heartbeat ghi `epoch x/200` mới là tiến độ train. Dòng `GPU utilization` chỉ cho biết GPU đang bận bao nhiêu phần trăm, **không phải tiến độ**.
'''),
    code(r'''
import re

LOG = Path('/kaggle/working/NB15_causal_uniform_seed42.log')
TAG = 'NB15-full'
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
print('Started A-causal-uniform seed 42 | PID', process.pid)
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
    print('GPU utilization below is load, not train progress:')
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
print('PASS: seed 42 training completed.')
'''),
    markdown('## 6. Chọn epoch và threshold bằng validation'),
    code(r'''
import pandas as pd

RUN_ROOT = repo / 'results_nb15_history_audit' / f'{CFG.stem}-{TAG}-gpu0'
RUN_DIR = RUN_ROOT / str(SEED)
SUMMARY = Path('/kaggle/working/summary_NB15_causal_uniform_validation_selected_seed42.csv')
subprocess.run([
    sys.executable, str(repo / 'scripts/summarize_thresholds.py'),
    str(RUN_ROOT), '--output', str(SUMMARY)], check=True)
selection = pd.read_csv(SUMMARY)
assert selection['seed'].astype(int).tolist() == [SEED]
best_epoch = int(selection.loc[0, 'best_epoch'])
threshold = float(selection.loc[0, 'threshold'])
checkpoint = RUN_DIR / 'ckpt' / f'{best_epoch}.ckpt'
assert checkpoint.exists(), f'Missing selected checkpoint: {checkpoint}'
SAFE_CHECKPOINT = Path('/kaggle/working/NB15_selected_seed42.ckpt')
SAFE_CONFIG = Path('/kaggle/working/AML-Small-HI-A15-causal-uniform-history-audit.yaml')
copy2(checkpoint, SAFE_CHECKPOINT)
copy2(CFG, SAFE_CONFIG)
assert SAFE_CHECKPOINT.exists() and SAFE_CHECKPOINT.stat().st_size > 0
display(selection)
print('Validation selected epoch:', best_epoch, '| threshold:', threshold)
print('Checkpoint backup saved before audit:', SAFE_CHECKPOINT)
'''),
    markdown('## 7. Xuất prediction và audit history/coverage trên validation'),
    code(r'''
AUDIT_DIR = Path('/kaggle/working/NB15_prediction_history_audit')
AUDIT_DIR.mkdir(exist_ok=True)
AUDIT_LOG = Path('/kaggle/working/NB15_prediction_history_audit.log')
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
print('PASS: full validation prediction/history audit completed.')
'''),
    markdown('## 8. Đọc kết quả đúng cách'),
    code(r'''
from IPython.display import Image, display

DETAIL = AUDIT_DIR / f'prediction_history_audit_{AUDIT_SPLIT}_seed{SEED}.csv'
BY_HISTORY = AUDIT_DIR / f'prediction_by_history_group_{AUDIT_SPLIT}_seed{SEED}.csv'
BY_COVERAGE = AUDIT_DIR / f'prediction_by_coverage_group_{AUDIT_SPLIT}_seed{SEED}.csv'
GLOBAL = AUDIT_DIR / f'prediction_audit_global_{AUDIT_SPLIT}_seed{SEED}.json'
FIGURE = AUDIT_DIR / f'prediction_history_audit_{AUDIT_SPLIT}_seed{SEED}.png'

history_result = pd.read_csv(BY_HISTORY)
coverage_result = pd.read_csv(BY_COVERAGE)
print('Kết quả toàn validation:')
print(GLOBAL.read_text())
print('\nTheo lượng lịch sử:')
display(history_result)
print('\nTheo sampling coverage:')
display(coverage_result)
display(Image(filename=str(FIGURE)))

print('\nQuy tắc quyết định:')
print('- Nếu history tăng, coverage giảm và Recall/F1 cũng giảm: giả thuyết history được ủng hộ → làm H-causal.')
print('- Nếu coverage giảm nhưng Recall/F1 không giảm: không được nói thiếu history gây lỗi → dừng hoặc đổi giả thuyết.')
print('- Đây là liên hệ thống kê, chưa phải bằng chứng nhân quả.')
'''),
    markdown('## 9. Đóng gói artifact để tải về'),
    code(r'''
import shutil

evidence = [
    CFG, SAFE_CHECKPOINT, SAFE_CONFIG, LOG, AUDIT_LOG, SUMMARY,
    DETAIL, BY_HISTORY, BY_COVERAGE, GLOBAL, FIGURE,
]
bundle = Path('/kaggle/working/NB15_prediction_history_audit_seed42_artifacts')
bundle.mkdir(exist_ok=True)
for path in evidence:
    if path.exists():
        shutil.copy2(path, bundle / path.name)
(bundle / 'commit.txt').write_text(commit + '\n', encoding='utf-8')
archive = shutil.make_archive(str(bundle), 'zip', bundle)
print('Download:', archive)
'''),
    markdown(r'''
## Sau Notebook 15

Chỉ nhìn hai bảng `prediction_by_history_group...csv` và `prediction_by_coverage_group...csv` để quyết định. Không chạy thêm ba seed ở bước khám phá này.

- Có chuỗi `history cao → coverage thấp → recall/F1 thấp`: thiết kế H-causal, khóa cấu hình rồi so sánh A-causal với H-causal trên seed 42–44.
- Không có chuỗi trên: không dùng history truncation làm lý do cải tiến; giữ kết quả temporal protocol làm đóng góp chính hoặc thu hẹp hướng đề tài.
'''),
]

notebook = {
    'cells': cells,
    'metadata': source.get('metadata', {
        'kernelspec': {
            'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3'},
    }),
    'nbformat': 4,
    'nbformat_minor': 5,
}
OUTPUT_NOTEBOOK.write_text(
    json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(OUTPUT_NOTEBOOK)
