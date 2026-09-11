from pathlib import Path
import json

NB_PATH = Path('tutorials/mitra_classifier_predictor_inference_colab.ipynb')
nb = json.loads(NB_PATH.read_text(encoding='utf-8'))
needle = "NETWORK_TIMEOUT_SECONDS = 30\n"
constants = (
    "MODEL_ID = 'autogluon/mitra-classifier'\n"
    "PINNED_REVISION = 'c425e9fa0910a6be1c494321792e7ba2a1367b1a'\n"
    "EXPECTED_WEIGHTS_SHA256 = 'e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19'\n"
    "EXPECTED_CONFIG_SHA256 = '2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df'\n"
)
changed = False
for cell in nb.get('cells', []):
    if cell.get('cell_type') != 'code':
        continue
    source = cell.get('source', [])
    text = ''.join(source) if isinstance(source, list) else str(source)
    if 'PIPELINE_API_REPO' not in text or 'PIPELINE_API = load_pinned_pipeline_api()' not in text:
        continue
    if constants not in text:
        if text.count(needle) != 1:
            raise SystemExit('inference setup constant insertion anchor mismatch')
        text = text.replace(needle, needle + "\n" + constants, 1)
        cell['source'] = text.splitlines(keepends=True)
    changed = True
    break
if not changed:
    raise SystemExit('could not locate inference production-API setup cell')
NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')

VALIDATOR = Path('scripts/validate_colab_tutorial.py')
text = VALIDATOR.read_text(encoding='utf-8')
anchor = "    code_text = \"\\n\".join(source for _, source in parsed_code)\n"
insert = '''    code_sequence = "\\n".join(source for _, source in parsed_code)\n    api_load_pos = code_sequence.find("PIPELINE_API = load_pinned_pipeline_api()")\n    require(api_load_pos >= 0, "inference tutorial must load the pinned production pipeline API")\n    for assignment in (\n        "MODEL_ID = 'autogluon/mitra-classifier'",\n        "PINNED_REVISION = 'c425e9fa0910a6be1c494321792e7ba2a1367b1a'",\n        "EXPECTED_WEIGHTS_SHA256 = 'e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19'",\n        "EXPECTED_CONFIG_SHA256 = '2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df'",\n    ):\n        assignment_pos = code_sequence.find(assignment)\n        require(\n            0 <= assignment_pos < api_load_pos,\n            f"inference tutorial must define {assignment.split(' = ', 1)[0]} before production-API compatibility checks",\n        )\n\n'''
# Insert in validate_inference_tutorial only: choose the anchor after the stale-package loop.
marker = "    for forbidden_text in (\n        \"DIMER ZIP\",\n        \"dimer-model-manifest.json\",\n        \"load_dimer_package\",\n    ):\n"
start = text.find(marker)
if start < 0:
    raise SystemExit('validator inference section marker missing')
pos = text.find(anchor, start)
if pos < 0:
    raise SystemExit('validator inference code_text anchor missing')
if 'inference tutorial must define {assignment.split' not in text:
    text = text[:pos] + insert + text[pos:]
VALIDATOR.write_text(text, encoding='utf-8')
