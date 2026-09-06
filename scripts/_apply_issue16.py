#!/usr/bin/env python3
import json
from pathlib import Path

MAIN = Path('tutorials/mitra_classifier_colab.ipynb')
INFERENCE = Path('tutorials/mitra_classifier_predictor_inference_colab.ipynb')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, nb):
    path.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def cell_text(cell):
    source = cell.get('source', [])
    return ''.join(source) if isinstance(source, list) else str(source)


def set_cell(cell, text):
    cell['source'] = text.splitlines(keepends=True)


def find_cell(nb, marker):
    matches = [cell for cell in nb['cells'] if marker in cell_text(cell)]
    if len(matches) != 1:
        raise SystemExit(f'Expected one cell containing {marker!r}; found {len(matches)}')
    return matches[0]


def replace_once(text, old, new, label):
    count = text.count(old)
    if count != 1:
        raise SystemExit(f'{label}: expected one occurrence, found {count}: {old!r}')
    return text.replace(old, new, 1)


main = load(MAIN)
for cell in main['cells']:
    text = cell_text(cell).replace('AUTOGLOUON_VERSION', 'AUTOGLUON_VERSION')
    set_cell(cell, text)

# Raw-header validation for every labelled CSV path before pandas can rename duplicates.
dataset = find_cell(main, 'def uploaded_csv')
text = cell_text(dataset)
text = replace_once(text, 'import io\n', 'import csv\nimport io\n', 'dataset csv import')
text = replace_once(
    text,
    "def uploaded_csv(payload):\n    return pd.read_csv(io.BytesIO(payload))\n",
    "def uploaded_csv(payload, label='uploaded CSV'):\n"
    "    # Check original names before pandas can rename duplicate headers.\n"
    "    text = payload.decode('utf-8-sig')\n"
    "    rows = csv.reader(io.StringIO(text, newline=''))\n"
    "    header = next((row for row in rows if row and not (len(row) == 1 and not row[0].strip())), [])\n"
    "    seen = set()\n"
    "    duplicates = []\n"
    "    for name in header:\n"
    "        if name in seen and name not in duplicates:\n"
    "            duplicates.append(name)\n"
    "        seen.add(name)\n"
    "    if duplicates:\n"
    "        raise ValueError(f'{label} contains duplicate column names: {duplicates}')\n"
    "    return pd.read_csv(io.BytesIO(payload))\n",
    'training CSV reader',
)
for old, new in (
    ("uploaded_csv(by_base['train.csv'])", "uploaded_csv(by_base['train.csv'], 'train.csv')"),
    ("uploaded_csv(by_base['val.csv'])", "uploaded_csv(by_base['val.csv'], 'val.csv')"),
    ("uploaded_csv(by_base['test.csv'])", "uploaded_csv(by_base['test.csv'], 'test.csv')"),
    ("train_data = pd.read_csv(z.open(names['train.csv']))", "train_data = uploaded_csv(z.read(names['train.csv']), 'train.csv')"),
    ("holdout_data = pd.read_csv(z.open(names['val.csv']))", "holdout_data = uploaded_csv(z.read(names['val.csv']), 'val.csv')"),
    ("test_data = pd.read_csv(z.open(names['test.csv']))", "test_data = uploaded_csv(z.read(names['test.csv']), 'test.csv')"),
):
    text = replace_once(text, old, new, 'training CSV path')
set_cell(dataset, text)

# Select between pretrained and fine-tuned predictors using holdout evidence only.
training = find_cell(main, "EVAL_METRIC = 'accuracy'")
text = cell_text(training)
text = replace_once(
    text,
    "MAX_MEMORY_USAGE_RATIO = 1.10      # @param {type:'number'}\n",
    "MAX_MEMORY_USAGE_RATIO = 1.10      # @param {type:'number'}\nMIN_SELECTION_HOLDOUT_ROWS = 50\n",
    'selection holdout constant',
)
text = replace_once(
    text,
    "    'active_predictor',\n",
    "    'active_predictor',\n    'recommended_predictor',\n    'active_mode',\n    'selection_basis',\n",
    'stale selection state',
)
selection = """

def metric_is_better(candidate, baseline, metric_name):
    if metric_name not in candidate or metric_name not in baseline:
        raise RuntimeError(f'Metric {metric_name!r} was not returned by AutoGluon; cannot select a predictor safely.')
    if metric_name == 'log_loss':
        return candidate[metric_name] < baseline[metric_name]
    return candidate[metric_name] > baseline[metric_name]


def metric_is_worse(candidate, baseline, metric_name):
    if metric_name not in candidate or metric_name not in baseline:
        return False
    if metric_name == 'log_loss':
        return candidate[metric_name] > baseline[metric_name]
    return candidate[metric_name] < baseline[metric_name]


recommended_predictor = baseline_predictor
active_mode = 'pretrained'
selection_basis = 'default:pretrained'
if finetuned_predictor is not None:
    if len(holdout_data) < MIN_SELECTION_HOLDOUT_ROWS:
        selection_basis = (
            f'default:pretrained; holdout-too-small:'
            f'{len(holdout_data)}<{MIN_SELECTION_HOLDOUT_ROWS}'
        )
        print(
            '⚠ Holdout is too small for automatic model selection '
            f'({len(holdout_data)} rows; minimum {MIN_SELECTION_HOLDOUT_ROWS}). '
            'Keeping the pretrained predictor for inference/export. '
            'Fine-tuned metrics are still reported as evaluation evidence.'
        )
    else:
        selection_basis = f'holdout:{EVAL_METRIC}'
        if metric_is_better(finetuned_metrics, baseline_metrics, EVAL_METRIC):
            recommended_predictor = finetuned_predictor
            active_mode = 'fine-tuned'
        print(
            f'✓ Recommended predictor for inference/export: {active_mode} '
            f'(selected by {EVAL_METRIC} on the holdout: '
            f'pretrained={baseline_metrics[EVAL_METRIC]:.6g}, fine-tuned={finetuned_metrics[EVAL_METRIC]:.6g}).'
        )
    if finetuned_test_metrics is not None and baseline_test_metrics is not None:
        degraded = [
            metric_name for metric_name in baseline_test_metrics
            if metric_name in finetuned_test_metrics
            and metric_is_worse(finetuned_test_metrics, baseline_test_metrics, metric_name)
        ]
        if degraded:
            details = ', '.join(
                f'{name}: {baseline_test_metrics[name]:.6g} → {finetuned_test_metrics[name]:.6g}'
                for name in degraded
            )
            print(
                '⚠ Fine-tuning produced mixed independent-test evidence. '
                f'These metrics worsened: {details}. Selection never uses the independent test; '
                'it remains evaluation evidence only.'
            )
else:
    print('✓ Recommended predictor for inference/export: pretrained (fine-tuning not run).')

active_predictor = recommended_predictor
"""
needle = "else:\n    print('Fine-tuning skipped. Set RUN_FINE_TUNING=True on a GPU to run it.')\n\nFIT_RUN_COMPLETED = True\n"
replacement = "else:\n    print('Fine-tuning skipped. Set RUN_FINE_TUNING=True on a GPU to run it.')\n" + selection + "\nFIT_RUN_COMPLETED = True\n"
text = replace_once(text, needle, replacement, 'selection block')
set_cell(training, text)

# Use the selected predictor consistently for inference and export.
for cell in main['cells']:
    text = cell_text(cell)
    if 'active = finetuned_predictor or baseline_predictor' in text:
        text = replace_once(text, 'active = finetuned_predictor or baseline_predictor', 'active = active_predictor', 'main inference active predictor')
    if 'active_predictor = finetuned_predictor or baseline_predictor' in text:
        text = replace_once(text, 'active_predictor = finetuned_predictor or baseline_predictor', 'active_predictor = recommended_predictor', 'export active predictor')
        text = replace_once(text, "    'mode': 'fine-tuned' if finetuned_predictor is not None else 'pretrained',\n", "    'mode': active_mode,\n    'selection_basis': selection_basis,\n", 'export selection metadata')
    set_cell(cell, text)
save(MAIN, main)

# Companion: verify archive before deserialization, assert task type, reject output collisions.
inference = load(INFERENCE)
for cell in inference['cells']:
    text = cell_text(cell).replace('AUTOGLOUON_VERSION', 'AUTOGLUON_VERSION')
    set_cell(cell, text)

trust = find_cell(inference, '## 2. Upload and validate `mitra-predictor.zip`')
text = cell_text(trust)
if 'Trust boundary' not in text:
    text += "\n**Trust boundary:** `TabularPredictor.load(...)` deserializes Python model objects. Load only a predictor ZIP you exported yourself or received from a trusted source. Path-safe extraction does not make an untrusted predictor archive safe to deserialize. If you saved the SHA-256 printed by the export step, paste it into `EXPECTED_ZIP_SHA256` so this notebook can verify the archive before loading it.\n"
set_cell(trust, text)

upload = find_cell(inference, "EXTRACT_ROOT = Path('/content/mitra-predictor-upload')")
text = cell_text(upload)
text = replace_once(text, "EXTRACT_ROOT = Path('/content/mitra-predictor-upload')\n", "EXPECTED_ZIP_SHA256 = ''  # @param {type:'string'}\nEXTRACT_ROOT = Path('/content/mitra-predictor-upload')\n", 'expected archive digest parameter')
verify_block = """zip_digest = sha256_file(ZIP_PATH)
expected_zip_digest = EXPECTED_ZIP_SHA256.strip().lower()
if expected_zip_digest:
    if len(expected_zip_digest) != 64 or any(ch not in '0123456789abcdef' for ch in expected_zip_digest):
        raise ValueError('EXPECTED_ZIP_SHA256 must be a 64-character hexadecimal SHA-256 digest.')
    if zip_digest != expected_zip_digest:
        raise RuntimeError(f'Predictor ZIP checksum mismatch. Expected {expected_zip_digest}; got {zip_digest}.')
    print('✓ Predictor ZIP SHA-256 matches the expected digest.')
else:
    print('⚠ No expected predictor ZIP digest supplied; continue only if this archive came from a trusted source.')
"""
text = replace_once(text, 'zip_digest = sha256_file(ZIP_PATH)\n', verify_block, 'archive digest verification')
set_cell(upload, text)

load_cell = find_cell(inference, 'predictor = TabularPredictor.load(str(PREDICTOR_ROOT))')
text = cell_text(load_cell)
text = replace_once(
    text,
    "predictor = TabularPredictor.load(str(PREDICTOR_ROOT))\n",
    "predictor = TabularPredictor.load(str(PREDICTOR_ROOT))\nif predictor.problem_type not in {'binary', 'multiclass'}:\n    raise RuntimeError(f'Expected a classification predictor, but loaded problem_type={predictor.problem_type!r}.')\n",
    'problem type assertion',
)
set_cell(load_cell, text)

predict_cell = find_cell(inference, 'probabilities = predictor.predict_proba(X, as_multiclass=True)')
text = cell_text(predict_cell)
collision_block = """probabilities = predictor.predict_proba(X, as_multiclass=True)

reserved_output_columns = ['prediction'] + [f'probability_{label}' for label in probabilities.columns]
output_collisions = [name for name in reserved_output_columns if name in new_data.columns]
if output_collisions:
    raise ValueError(
        f'Inference CSV contains output column(s) reserved by this notebook: {output_collisions}. '
        'Rename or remove them before inference.'
    )
"""
text = replace_once(text, 'probabilities = predictor.predict_proba(X, as_multiclass=True)\n', collision_block, 'output collision guard')
set_cell(predict_cell, text)
save(INFERENCE, inference)
