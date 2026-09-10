from __future__ import annotations

import json
from pathlib import Path

NOTEBOOK = Path("tutorials/mitra_classifier_colab.ipynb")
README = Path("tutorials/README.md")
SOURCE_REVISION = "8e766925733b2726cc5c1c74cc3605e35f76b811"

nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))

def text(cell: dict) -> str:
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src

def put(cell: dict, value: str) -> None:
    cell["source"] = value

cell = next(
    c for c in nb["cells"]
    if c.get("cell_type") == "code"
    and "DATA_SOURCE = 'Sample dataset (FreshRetailNet)'" in text(c)
    and "def prepare(" in text(c)
)
src = text(cell)
src = src.replace(
    "DATA_SOURCE = 'Sample dataset (FreshRetailNet)'  # @param ['Sample dataset (FreshRetailNet)', 'Upload CSV', 'Upload pre-split train/val/test']",
    "DATA_SOURCE = 'Sample: FreshRetailNet (temporal demand)'  # @param ['Sample: FreshRetailNet (temporal demand)', 'Sample: Telco Customer Churn', 'Sample: Adult Census Income', 'Sample dataset (FreshRetailNet)', 'Upload CSV', 'Upload pre-split train/val/test']",
)
old_sample_defs = """SAMPLE_REVISION = '8fc19e80ae3166ec6bf964d194a28c80e6ba3b1f'\nSAMPLE_ZIP_URL = f'https://raw.githubusercontent.com/kurtvalcorza/mitra-classifier-pipeline/{SAMPLE_REVISION}/examples/sample-data/freshretailnet-band-h7.zip'\nSAMPLE_CARD_URL = f'https://github.com/kurtvalcorza/mitra-classifier-pipeline/blob/{SAMPLE_REVISION}/examples/sample-data/DATASET_CARD.md'\n\nusing_sample = DATA_SOURCE == 'Sample dataset (FreshRetailNet)'\nusing_presplit_data = DATA_SOURCE in {'Sample dataset (FreshRetailNet)', 'Upload pre-split train/val/test'}\n"""
new_sample_defs = f"""SAMPLE_REVISION = '{SOURCE_REVISION}'\nSAMPLE_BASE_URL = f'https://raw.githubusercontent.com/kurtvalcorza/mitra-classifier-pipeline/{{SAMPLE_REVISION}}/examples/sample-data'\nSAMPLE_CARD_URL = f'https://github.com/kurtvalcorza/mitra-classifier-pipeline/blob/{{SAMPLE_REVISION}}/examples/sample-data/DATASET_CARD.md'\nSAMPLE_CONFIGS = {{\n    'Sample: FreshRetailNet (temporal demand)': {{'file': 'freshretailnet-band-h7.zip', 'target': 'target'}},\n    'Sample dataset (FreshRetailNet)': {{'file': 'freshretailnet-band-h7.zip', 'target': 'target'}},\n    'Sample: Telco Customer Churn': {{'file': 'telco-customer-churn.zip', 'target': 'Churn'}},\n    'Sample: Adult Census Income': {{'file': 'adult-census-income.zip', 'target': 'class'}},\n}}\n\nusing_sample = DATA_SOURCE in SAMPLE_CONFIGS\nusing_presplit_data = using_sample or DATA_SOURCE == 'Upload pre-split train/val/test'\nif using_sample:\n    sample_cfg = SAMPLE_CONFIGS[DATA_SOURCE]\n    TARGET_COLUMN = sample_cfg['target']\n    SAMPLE_ZIP_URL = f\"{{SAMPLE_BASE_URL}}/{{sample_cfg['file']}}\"\n"""
if old_sample_defs not in src:
    raise RuntimeError("Could not locate current hardened sample definitions")
src = src.replace(old_sample_defs, new_sample_defs)

# Sample target must be known before DROP_COLUMNS is interpreted.
marker = "drop_columns = [c.strip() for c in DROP_COLUMNS.split(',') if c.strip() and c.strip() != TARGET_COLUMN]"
if marker not in src:
    raise RuntimeError("Could not locate DROP_COLUMNS normalization")
if src.index("if using_sample:\n    sample_cfg") > src.index(marker):
    raise RuntimeError("Sample target resolution did not precede DROP_COLUMNS")

put(cell, src)

step3 = next(c for c in nb["cells"] if c.get("cell_type") == "markdown" and text(c).startswith("## 3."))
md = text(step3)
if "Telco Customer Churn" not in md:
    insert = (
        "\n\n**Sample portfolio.** The default FreshRetailNet temporal classification sample remains available, "
        "including its legacy selector. `Sample: Telco Customer Churn` adds a mixed-feature binary "
        "classification task and `Sample: Adult Census Income` adds a demographic binary classification "
        "task. The additional archives are pinned to an immutable repository revision and retain their "
        "provided train/validation/test splits. They are tutorial/sanity datasets, not benchmark evidence. "
        "See `examples/sample-data/DATASET_CARD.md` for provenance, schema, split construction, and licenses.\n"
    )
    first_break = md.find("\n\n", len("## 3."))
    md = md[:first_break] + insert + md[first_break:]
put(step3, md)

NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

readme = README.read_text(encoding="utf-8")
if "## Sample dataset portfolio" not in readme:
    readme += f'''\n\n## Sample dataset portfolio\n\nThe hardened E2E classifier tutorial exposes three sample paths:\n\n- **FreshRetailNet** — temporal demand-band classification with preserved leakage-aware splits.\n- **Telco Customer Churn** — mixed-feature binary classification (`Churn`).\n- **Adult Census Income** — mixed-feature binary classification (`class`).\n\nThe sample archives and dataset card are pinned to repository revision `{SOURCE_REVISION}`; the Notebook Specification v1.0 validation, BYOD, model-integrity, artifact, and inference boundaries remain unchanged.\n'''
    README.write_text(readme, encoding="utf-8")
