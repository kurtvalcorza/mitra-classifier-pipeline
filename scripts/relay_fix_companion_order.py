#!/usr/bin/env python3
import json
from pathlib import Path

path = Path('tutorials/mitra_classifier_predictor_inference_colab.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
anchor = "PIPELINE_API_REPO = 'kurtvalcorza/mitra-classifier-pipeline'"
constants = (
    "MODEL_ID = 'autogluon/mitra-classifier'\n"
    "PINNED_REVISION = 'c425e9fa0910a6be1c494321792e7ba2a1367b1a'\n"
    "EXPECTED_WEIGHTS_SHA256 = 'e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19'\n"
    "EXPECTED_CONFIG_SHA256 = '2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df'\n\n"
)
changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    text = ''.join(source) if isinstance(source, list) else str(source)
    if anchor not in text:
        continue
    before_anchor = text.split(anchor, 1)[0]
    if "MODEL_ID = 'autogluon/mitra-classifier'" not in before_anchor:
        text = text.replace(anchor, constants + anchor, 1)
        cell['source'] = text.splitlines(keepends=True)
        changed = True
    break
else:
    raise SystemExit('Could not locate companion pipeline API initialization cell')

if not changed:
    raise SystemExit('Companion initialization constants already present; refusing no-op relay commit')

path.write_text(json.dumps(nb, indent=1) + '\n', encoding='utf-8')
