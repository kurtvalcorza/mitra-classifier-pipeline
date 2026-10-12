"""Generate tutorials/DIMER_Philippines_Rice_Pixel_Tables_Workshop.ipynb (deterministic; `--check` for parity).

Spec: docs/RICE_TABULAR_WORKSHOP_SPEC.md (owner decisions D1-D8 in section 13). The notebook carries, hash-checked:
tools/rice_workshop/rice_tables.py (kernel module) and tools/rice_workshop/fm_runner.py (isolated-environment runner),
plus the v2 workshop's environment cell (adapters, model manifests, hash locks, pinned uv) verbatim up to its
condition-selection tail, read from the v2 notebook in this repository at generation time.
"""

# ruff: noqa: E501  (learner-facing markdown is kept as one line per paragraph)
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NOTEBOOK = REPO / "tutorials" / "DIMER_Philippines_Rice_Pixel_Tables_Workshop.ipynb"
V2 = REPO / "tutorials" / "DIMER_FreshRetailNet_MultiModel_Classification_Workshop_v2.ipynb"
GENERATOR = "tools/rice_workshop/build_workshop.py"
WORKSHOP_REVISION = "1.0.0"
NOTEBOOK_SPEC = "2.4"
PIECE = 1000  # at most this many characters per carried string piece, so no cell line is unreadably long
V2_CELL_TITLE = "# @title 5.2 Stage carried adapters and prepare isolated model environments"
V2_CUT = "selected_environment_keys = sorted("


def _read(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def v2_environment_head() -> str:
    """The v2 workshop's section 5.2 cell up to its condition-selection tail, byte for byte (the smoke used the same
    cut on a Colab T4). It defines the carried adapters, model manifests, locks, pinned uv and ensure_environment()."""
    notebook = json.loads(_read(V2))
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith(V2_CELL_TITLE):
            head = source[: source.index(V2_CUT)]
            if "def ensure_environment(environment_key):" not in head:
                raise RuntimeError("the v2 environment cell changed shape; review the carriage")
            return head
    raise RuntimeError(f"{V2.name}: section 5.2 cell not found")


def carried_files() -> dict[str, str]:
    return {name: _read(HERE / name) for name in ("rice_tables.py", "fm_runner.py")}


def _literal(text: str) -> str:
    pieces: list[str] = []
    for line in text.splitlines(keepends=True):
        while len(line) > PIECE:
            pieces.append(line[:PIECE])
            line = line[PIECE:]
        pieces.append(line)
    return "(\n" + "".join(f"        {piece!r}\n" for piece in pieces) + "    )"


# --------------------------------------------------------------------------------------------------
# opening
# --------------------------------------------------------------------------------------------------

TITLE = r"""# Rice Pixels as a Table: Better Features or Better Models?

**DIMER Philippines Rice Pixel Tables Workshop** · **Profile:** `E2E` · **Mode:** `WORKSHOP` · **Kind:** `WORKSHOP` · **Notebook standard:** DIMER Notebook Specification `2.4`, standalone · **Release status:** Candidate · **Data:** Kaggle `kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching`, version 1

> **⚠️ Notice.** For research, training and evaluation only. Not validated for production, clinical, diagnostic or safety-critical use. Running this notebook downloads third-party model weights and datasets under their own licenses (see the model card) and uses compute on your own Colab or Kaggle account. Data you bring stays in your runtime, and you are responsible for having the right to use it. Results here are tutorial evidence, not benchmarks. Everything is provided "as is", without warranty of any kind. To the maximum extent permitted by law, the maintainers and the DIMER platform accept no liability for its use. The rice maps it produces are not official crop statistics and must not be used for agricultural insurance, subsidy or food-security decisions.

> **TabPFN-3 licence.** The TabPFN-3 weights (`Prior-Labs/tabpfn_3`) are released under the TabPFN-3 licence v1.0 (**non-commercial weights**), which does not permit commercial use. This notebook uses them for teaching and research only; do not deploy the TabPFN-3 conditions it fits, or their outputs, commercially. The other weights carry their own licences: TabICLv2 BSD-3-Clause, TabDPT v1.2 Apache-2.0, Mitra Apache-2.0 (table in **Data, models, attribution and licences** at the end).

## Problem statement

Rice-area maps guide planting advisories, irrigation planning and harvest forecasts in the Philippines. Mapping rice from satellites usually means turning each 20 m pixel into a row of numbers and classifying it. Two things can improve such a map: a better **description of each pixel** (the features) or a better **classifier**. Tabular foundation models — models pretrained on many tables that classify a new table *in context* — promise better classifiers without tuning. This notebook asks, on the same Philippine rice pixels: **how much does the feature set change rice-mapping quality compared with the model, and what can 8 validation chips actually establish?**

A useful outcome is a rice map with high pooled **rice F1** (the balance of precision and recall on the rice class) on chips the models never saw, measured under one frozen protocol. The notebook does not produce a national rice map, does not compare against official statistics, and makes no claim beyond the two validation regions (Central and Eastern Visayas) it scores.

**Who this notebook is for.** Learners who can open a hosted notebook, run cells in order and read short Python, and who have met rice mapping in course lessons L3 (Sentinel-2 + LightGBM) and L3B (AlphaEarth + LightGBM). No deep-learning experience is needed. Terms are explained at first use and collected in the glossary.

**Prerequisites and runtime.** A fresh Google Colab session with a **T4 GPU** (Runtime → Change runtime type → T4 GPU), Linux x86_64, about 30 GB free disk. No account, token, upload or DIMER service is needed. The notebook downloads the public teaching dataset (about 344 MB), three model checkpoints (about 0.6 GB together) and builds three isolated, hash-locked Python 3.12.12 environments of about 5–6 GB each (a fourth, about 7.5 GB, only if you switch Mitra on). **Expected runtime: about 30–40 minutes** for the default path on a fresh Colab T4 (an estimate from development runs on a Tesla T4, 12.7 GB RAM, on 2026-10-11; this run records its own times).

**Input → System → Output.** One labelled 20 m pixel from a 2.56 km chip-year → one of two **feature sets** (Sentinel-2 **M1**: 70 spectral-temporal summary values; or **AlphaEarth**: a 64-value learned embedding plus its validity flag) → one of five **systems**: A, call everything non-rice; B, LightGBM [6]; C1 TabPFN-3 [1], [2]; C2 TabICLv2 [3]; C4 TabDPT v1.2 [4] (C3 Mitra [5] is opt-in) → a rice probability per pixel, a threshold chosen on development pixels, and a 0/1 rice decision scored once against a published rice map [9].

### Learning objectives

By the end of this notebook you will be able to:

1. **Build** a pixel table from image chips without letting a chip leak between fit, development and validation roles.
2. **Explain** what the 70-value M1 summary and the 64-value AlphaEarth embedding each encode.
3. **Explain** in-context learning — fitting registers support rows, with no gradient step — and **predict** how context size affects it.
4. **Compare** five systems on two feature sets under one frozen protocol, and **separate** a feature effect from a model effect.
5. **Interpret** pooled precision, recall, F1 and IoU for a minority class, and **diagnose** per-chip failures.
6. **Write** a conclusion that states its evidence and what 8 clustered validation chips cannot support.

### How to use this notebook

- **Run all** from a fresh T4 runtime (Runtime → Run all). Every cell runs in order without editing; nothing asks for input and no restart is needed. Mitra and bring-your-own-data are switched off by default.
- Cells titled **Infrastructure** are collapsed. They make the run reproducible; you may run them without reading their code.
- Sections are tagged **[Concept]**, **[Evaluation practice]** or **[Engineering]**.
- Before each comparison you are asked to **predict**. Write your prediction down first. After each result, **What to notice** describes normal output, and a collapsed **Check your reasoning** box holds a sample answer: open it after answering.
- Form fields (`# @param`) are the only values you change, and only in the marked activity and optional sections.

### Roadmap

| Section | What you do | Kind |
|---|---|---|
| 0 | Prepare the runtime and the notebook's own helper code | Engineering |
| 1 | Download, verify and featurise the chips; look at the tables | Concept |
| 2 | Predict, then run baselines A and B; choose thresholds on development | Evaluation practice |
| 3 | Run the tabular foundation models in isolated environments | Concept |
| 4 | Compare on development: the 2 × 5 grid | Evaluation practice |
| 5 | Activity: change the context size of one model | Evaluation practice |
| 6 | Freeze every choice | Evaluation practice |
| 7 | Score validation once and interpret | Evaluation practice |
| 8 | Reload check and export `results.zip` | Engineering |
| 9 | Conclude; what did not work | Evaluation practice |
| 10 | Optional: bring your own chips (re-adapts every system) | Engineering |

**Fast path.** If time is short, run everything and read sections 4, 5 and 7; the other sections explain how the evidence was produced.

**Evidence boundary.** 40 chip-years: 32 for training (24 fit, 8 development), 8 for validation, all validation chips from two regions. This is a teaching comparison, not a model ranking and not a national accuracy estimate. Course results for orientation (lesson evidence, not this run): L3 Sentinel-2 LightGBM reached rice F1 0.760 and L3B AlphaEarth LightGBM 0.928 on the lessons' teaching split."""

DATA_CONTEXT = r"""## Data context [Concept]

**What it is.** The Kaggle dataset `kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching` [10], version 1 (CC BY 4.0), holds 40 chip-years: 128 × 128 pixels at 20 m (2.56 km) across Philippine regions, years 2019–2021. Each HDF5 chip holds:

| Dataset | Shape | Meaning |
|---|---|---|
| `S2` | 24 × 6 × 128 × 128 | Sentinel-2 [8] surface reflectance, 24 calendar bins (about half-monthly medians) × 6 bands (B2 blue, B3 green, B4 red, B8A narrow NIR, B11 and B12 short-wave infrared) |
| `S2_valid` | 24 × 128 × 128 | 1 where a bin had a clear observation, 0 where clouds or gaps removed it |
| `AlphaEarth` | 64 × 128 × 128 | AlphaEarth Foundations annual satellite embedding [7]: 64 learned values per pixel summarising a year of observations from several sensors |
| `AE_valid` | 128 × 128 | 1 where the embedding exists |
| `label` | 128 × 128 | 0 non-rice, 1 rice, 255 ignored (edges and unmapped) |

**Who produced it and how.** The rice labels are a published 20 m rice map of Monsoon Asia [9], reprojected and binarised; the optical inputs are modified Copernicus Sentinel data; the embeddings come from Google's public AlphaEarth product; the compilation is by Kurt Valcorza. Full attribution is at the end.

**How it relates to the problem.** The chips sample rice and non-rice landscapes, not the country evenly. All 8 validation chips are in two regions (PH-07, PH-08), two of them contain no rice, and neighbouring pixels are strongly alike, so pixel counts overstate how much independent evidence there is [11]. AlphaEarth was pretrained on global imagery that may include these places and years, so its advantage may partly reflect having seen them. The labels are themselves a model's output, not field surveys.

**Roles.** The capstone's rule splits the 32 training chips by **whole region** into 24 fit and 8 development chips (salted SHA-256 order, about 25% development, at least two rice chips in development), so this notebook and the Prithvi rice capstone score identical chips. The `development-example` chip is a training sample and is never read. Validation pixels are scored once, after every choice is frozen.

**Rows of the table.** One row per labelled pixel. The **fit table** is a class-stratified sample of 10,000 pixels from the fit chips (3,500 or so rice) — the same rows for every foundation model — plus a 65,536-row table for LightGBM. Choices are made on a seeded 30,000-pixel **development subset** that keeps the development rice share. Validation uses **every** labelled validation pixel.

<details><summary><b>Glossary</b> (open when a term is unfamiliar)</summary>

| Term | Meaning here |
|---|---|
| Chip, chip-year | A 128 × 128-pixel square for one year |
| Feature set | The numbers that describe a pixel: M1 (70) or AlphaEarth (65) |
| M1 features | Per band: mean, standard deviation, minimum, maximum, 10th/50th/90th percentile of the clear observations; per quarter: each band's mean and the share of clear bins |
| Embedding | A learned vector that summarises an input; nearby vectors mean similar places |
| In-context learning | A pretrained model classifies new rows by attending to labelled support rows given at prediction time; "fit" stores those rows and trains nothing |
| Context size | How many labelled rows the in-context model sees |
| Fit / development / validation | Rows the model learns from / rows used to choose settings / rows scored once at the end |
| Threshold | The probability above which a pixel is called rice |
| Precision, recall | Of pixels called rice, the share that are rice; of rice pixels, the share found |
| F1, IoU | 2PR/(P+R); rice overlap ÷ rice union. Both ignore true negatives, so they suit a minority class [12] |
| Pooled | Counted over all pixels together, not averaged per chip |
| Freeze | Recording every choice, with hashes, before looking at validation |
| Artifact | The saved, reloadable fitted model and its context |
</details>"""

# --------------------------------------------------------------------------------------------------
# section 0
# --------------------------------------------------------------------------------------------------

SEC0 = r"""# 0. Set up the runtime [Engineering]

**Infrastructure — run the cell; reading its implementation is optional.** It checks that the runtime is Linux x86_64 with the packages Colab already ships (numpy, pandas, matplotlib, LightGBM, h5py, kagglehub), creates a fresh session folder, writes the notebook's own helper code to it after checking each file's SHA-256, and imports it. Nothing is installed into this notebook's Python."""


def cell_setup(files: dict[str, str]) -> str:
    hashes = {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in files.items()}
    lines = [
        "# @title 0.1 Infrastructure: runtime check, session folder and the notebook's helper code",
        "from __future__ import annotations",
        "",
        "import gc",
        "import hashlib",
        "import importlib.util",
        "import json",
        "import os",
        "import platform",
        "import shutil",
        "import subprocess",
        "import sys",
        "import threading",
        "import time",
        "import urllib.request",
        "import uuid",
        "import zipfile",
        "from pathlib import Path",
        "",
        "if platform.system() != 'Linux' or platform.machine() != 'x86_64':",
        "    raise RuntimeError('This notebook needs a Linux x86_64 runtime (Google Colab, Kaggle or Linux Jupyter); '",
        "                       f'this kernel reports {platform.system()} {platform.machine()}.')",
        "missing = [n for n in ('numpy', 'pandas', 'matplotlib', 'lightgbm', 'h5py', 'kagglehub')",
        "           if importlib.util.find_spec(n) is None]",
        "if missing:",
        "    raise RuntimeError('This kernel lacks ' + ', '.join(missing) + '. Colab includes them; on another Linux '",
        "                       'Jupyter server add them to the kernel environment before starting it.')",
        "",
        "import lightgbm",
        "import matplotlib.pyplot as plt",
        "import numpy as np",
        "import pandas as pd",
        "from IPython.display import FileLink, Markdown, display",
        "",
        f"WORKSHOP_REVISION = {WORKSHOP_REVISION!r}",
        "RANDOM_SEED = 42",
        "SESSION_STARTED = time.perf_counter()",
        "ROOT = Path('/content') if Path('/content').exists() else Path.cwd()",
        "WORKSPACE_ROOT = ROOT / 'dimer_rice_pixel_tables'",
        "CACHE_ROOT = WORKSPACE_ROOT / 'cache'",
        "ENVIRONMENT_ROOT = WORKSPACE_ROOT / 'environments'  # reused across sessions when its identity matches",
        "SESSION_ROOT = WORKSPACE_ROOT / 'sessions' / uuid.uuid4().hex[:12]  # every run of this cell starts clean",
        "RUN_ROOT = SESSION_ROOT / 'runs'",
        "SUPPORT_ROOT = SESSION_ROOT / 'support'",
        "REBUILD_MODEL_ENVIRONMENTS = False",
        "for directory in (WORKSPACE_ROOT, CACHE_ROOT, ENVIRONMENT_ROOT, SESSION_ROOT, RUN_ROOT, SUPPORT_ROOT):",
        "    directory.mkdir(parents=True, exist_ok=True)",
        "",
        "def sha256_file(path):",
        "    digest = hashlib.sha256()",
        "    with Path(path).open('rb') as handle:",
        "        for block in iter(lambda: handle.read(1 << 20), b''):",
        "            digest.update(block)",
        "    return digest.hexdigest()",
        "",
        "def canonical_json(value):",
        "    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)",
        "",
        "def fingerprint(value):",
        "    return hashlib.sha256(canonical_json(value).encode('utf-8')).hexdigest()",
        "",
        "CARRIED_FILES = {",
    ]
    lines += [f"    {name!r}: {_literal(text)}," for name, text in files.items()]
    lines += ["}", "CARRIED_HASHES = {"]
    lines += [f"    {name!r}: {digest!r}," for name, digest in hashes.items()]
    lines += [
        "}",
        "for name, text in CARRIED_FILES.items():",
        "    data = text.encode('utf-8')",
        "    if hashlib.sha256(data).hexdigest() != CARRIED_HASHES[name]:",
        "        raise RuntimeError(f'{name}: embedded text does not match its pinned SHA-256; regenerate the notebook')",
        "    (SUPPORT_ROOT / name).write_bytes(data)",
        "module_spec = importlib.util.spec_from_file_location('rice_tables', SUPPORT_ROOT / 'rice_tables.py')",
        "rt = importlib.util.module_from_spec(module_spec)",
        "sys.modules['rice_tables'] = rt",
        "module_spec.loader.exec_module(rt)",
        "FM_RUNNER = SUPPORT_ROOT / 'fm_runner.py'",
        "",
        "def gpu_description():",
        "    try:",
        "        return subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],",
        "                              capture_output=True, text=True, timeout=30).stdout.strip() or 'none detected'",
        "    except (OSError, subprocess.SubprocessError) as exc:",
        "        return f'none detected ({type(exc).__name__})'",
        "",
        "HOST = {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__,",
        "        'lightgbm': lightgbm.__version__, 'gpu': gpu_description(),",
        "        'disk_free_gb': round(shutil.disk_usage(WORKSPACE_ROOT).free / 1e9, 1)}",
        "display(pd.Series(HOST, name='this runtime').to_frame())",
        "if 'none detected' in HOST['gpu']:",
        "    print('No GPU detected: the foundation models will run on CPU and take hours. Select Runtime > Change runtime '",
        "          'type > T4 GPU, then Runtime > Run all.')",
        "print(f'Session folder: {SESSION_ROOT}')",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------------------------------
# section 1: data
# --------------------------------------------------------------------------------------------------

SEC1 = r"""# 1. Download, verify and featurise the chips [Concept]

The next cell downloads version 1 of the dataset anonymously with `kagglehub` (on Kaggle it uses the attached copy), checks `INDEX.jsonl` against its pinned SHA-256 and every chip against the SHA-256 in the index, assigns fit, development and validation roles by whole region, and turns every labelled pixel into two rows: one of 70 M1 values, one of 65 AlphaEarth values. Each chip's datasets are checked for shape, type, value range and label values before use; `label_raw` is never read.

It then writes the tables the models use: the 10,000-row fit sample, LightGBM's 65,536-row sample, the 30,000-pixel development subset and all labelled validation pixels. **Missing values** — an M1 statistic with no clear observation, or an AlphaEarth pixel without an embedding — are filled with the column mean of the model's own fit sample, never with development or validation values.

This cell takes about 6–7 minutes on a T4 runtime: the M1 percentiles are computed for 655,360 pixels."""

CELL_PREPARE = r"""# @title 1.1 Download, verify, assign roles and stage the pixel tables
LEDGER = rt.Ledger(RUN_ROOT / 'sample')
DATA_SOURCE = rt.download_dataset(CACHE_ROOT)
PREPARED = rt.prepare_sample(LEDGER, DATA_SOURCE, CACHE_ROOT)
CHIPS = pd.DataFrame(PREPARED['chips'])
display(CHIPS.groupby('role').agg(chips=('chip', 'count'), regions=('region', lambda r: ', '.join(sorted(set(r)))),
                                  labelled_pixels=('labelled_pixels', 'sum')))
TABLE_SIZES = pd.DataFrame([
    {'feature_set': fs, 'table': key, 'rows': v['rows'], 'rice': v['rice'] if key != 'eval' else 'hidden until scored'}
    for fs, rec in PREPARED['tables']['feature_sets'].items() for key, v in rec.items() if isinstance(v, dict)])
display(TABLE_SIZES)
print(f"INDEX and {len(CHIPS)} chips verified against their SHA-256 values; layout: {PREPARED['data_manifest']['layout']}; "
      f"{LEDGER.receipt('prepare')['seconds']} s")"""

AFTER_PREPARE = r"""**What to notice:** 24 fit, 8 development and 8 validation chips, with no region in two roles. The fit table has 10,000 rows for both feature sets, roughly a third of them rice because the sample takes the same number of pixels from each chip and class where it can; the development subset keeps the development chips' own rice share. Validation rice counts are hidden until the scores are in, so they cannot steer a choice."""

SEC1_LOOK = r"""## 1.2 Look at the two feature sets [Concept]

Rice is a temporal signal: paddies are flooded (low near-infrared, low NDVI), green up, then are harvested. M1 keeps a little of that timing through its quarterly band means. AlphaEarth packs a year of several sensors into 64 numbers that are not individually interpretable. The next cell plots the quarterly NDVI, computed from M1's quarterly red and near-infrared means, for rice and non-rice fit pixels, and lists how many AlphaEarth values are missing."""

CELL_LOOK = r"""# @title 1.2 Quarterly NDVI by class, and missing embeddings (fit rows only)
fit_x, fit_y = rt.load_split(LEDGER.root / 'tables', 's2m1', 'fit')
names = rt.FEATURE_SETS['s2m1']
quarters = range(1, rt.QUARTERS + 1)
ndvi = {q: (fit_x[:, names.index(f'q{q}_B8A_mean')] - fit_x[:, names.index(f'q{q}_B4_mean')])
           / (fit_x[:, names.index(f'q{q}_B8A_mean')] + fit_x[:, names.index(f'q{q}_B4_mean')] + 1e-6) for q in quarters}
fig, ax = plt.subplots(figsize=(6, 3.2))
for label, name in ((1, 'rice'), (0, 'non-rice')):
    ax.plot(list(quarters), [np.median(ndvi[q][fit_y == label]) for q in quarters], marker='o', label=name)
ax.set_xticks(list(quarters), [f'Q{q}' for q in quarters])
ax.set_ylabel('median NDVI'); ax.set_title('Fit pixels: quarterly NDVI from M1 features'); ax.legend()
plt.tight_layout(); plt.show()
ae_x, _ = rt.load_split(LEDGER.root / 'tables', 'alphaearth', 'dev')
print(f"Development subset: {np.isnan(ae_x[:, 0]).mean():.1%} of pixels have no AlphaEarth embedding (AE_valid = 0); "
      'they are filled with fit means and flagged by the AE_valid column.')
display(pd.DataFrame(fit_x[:3, :8], columns=names[:8]).round(3))
del fit_x, ae_x; gc.collect()"""

AFTER_LOOK = r"""**What to notice:** the rice curve usually differs from non-rice in at least one quarter (a dip at flooding or a peak at full canopy); a quarter where the two curves overlap carries little signal. M1 values are readable reflectance statistics, AlphaEarth's are not: a model has to discover what each embedding axis means from the fit rows.

**Checkpoint.** Which feature set do you expect to separate rice better, and why might the answer differ between a gradient-boosted tree model and an in-context model?

<details><summary>Check your reasoning</summary>AlphaEarth summarises many observations and sensors with a model trained on far more imagery, so it may separate classes better with any classifier. M1 is limited to six bands and four quarters but is grounded in physics. A tree model splits on individual columns; an in-context model compares whole rows with its support rows, so it may use correlated embedding axes differently. Section 4 tests this.</details>"""

# --------------------------------------------------------------------------------------------------
# section 2: baselines
# --------------------------------------------------------------------------------------------------

SEC2 = r"""# 2. Predict, then run the baselines [Evaluation practice]

**The question.** On the same pixels, which matters more for rice F1: switching from M1 to AlphaEarth, or switching from LightGBM to a tabular foundation model? **Write down your prediction now**, as two rough numbers: the F1 gain from changing the feature set, and the spread between the best and worst model on one feature set.

**The protocol** (fixed before any model runs):

1. Roles come from whole regions; every fit sample and fill uses fit chips only.
2. Every choice is made on the development subset: each condition's **threshold** (0.10 to 0.90 in steps of 0.05, highest pooled development rice F1, ties nearest 0.5) and LightGBM's number of trees (early stopping on development F1).
3. Every choice and every condition's identity is written to `selection.json` and hashed (the **freeze**). Only then is validation scored, once, and the result is kept whichever system wins.

**Baselines.** A calls every pixel non-rice: its accuracy equals the non-rice share, and its recall and F1 are 0. B is LightGBM [6] with the capstone's settings (31 leaves, learning rate 0.05, at most 200 trees, patience 50) on 65,536 fit rows. B10k is the same model on the 10,000 rows the foundation models get, so it compares LightGBM with them on identical data. The two LightGBM tables are drawn separately, so B versus B10k differs in which rows as well as how many: read it as a rough sense of how much more data helps, not as a pure sample-size effect."""

CELL_BASELINE = r"""# @title 2.1 Baselines A, B and B10k on both feature sets (development only)
BASELINE = rt.baseline_stage(LEDGER)
display(pd.DataFrame([{'condition': k, 'fit rows': v.get('fit_rows', '-'), 'trees': v.get('best_iteration', '-'),
                       'dev F1 at 0.5': v['dev_f1_at_0p5'], 'chosen threshold': v['threshold'],
                       'dev F1 at chosen': v['dev_f1_selected'], 'seconds': v['seconds']}
                      for k, v in BASELINE.items()]).round(4))"""

AFTER_BASELINE = r"""**What to notice:** A's F1 is 0 on both feature sets. B on AlphaEarth is usually well above B on M1; the chosen threshold is often near 0.5 but need not be. B10k usually trails B slightly: fewer rows give the trees less to learn from (the rows also differ, so part of the gap can be the particular sample). These are development numbers, used for choosing, not for claiming."""

# --------------------------------------------------------------------------------------------------
# section 3: foundation models
# --------------------------------------------------------------------------------------------------

SEC3 = r"""# 3. Tabular foundation models [Concept]

A **tabular foundation model** is a neural network pretrained on millions of synthetic or real tables to predict a row's label from other labelled rows. Given a new table, it does not train: `fit` stores the 10,000 labelled fit rows as **context**, and `predict` passes each new row through the network together with that context. This is **in-context learning**. The trade-offs: there is no training loop to tune, but every prediction pays for attending to the context, the context has a size limit, and the model's pretraining decides what patterns it can recognise.

| ID | System | Checkpoint (pinned revision) | Licence | Settings here |
|---|---|---|---|---|
| C1 | TabPFN-3 [1], [2] | `Prior-Labs/tabpfn_3` @ `24a16a89d245` | TabPFN-3 licence v1.0, **non-commercial** | 4 estimators |
| C2 | TabICLv2 [3] | `jingang/TabICL` @ `4dcd344ece2c` | BSD-3-Clause | 8 estimators |
| C3 | Mitra [5] (opt-in, off) | `autogluon/mitra-classifier` @ `c425e9fa0910` | Apache-2.0 | AutoGluon 1.5.0, in-context |
| C4 | TabDPT v1.2 [4] | `Layer6/TabDPT` @ `4462ffbd1d8d` | Apache-2.0 | 4 ensembles, 4,096-row retrieved context |

Every model gets the same 10,000 fit rows. TabDPT retrieves the 4,096 most similar of them for each prediction. Each model runs in its own process of its own environment, one at a time, so GPU memory is released between models.

**Mitra is off by default.** On a T4 it predicts about 50 rows per second, so one full validation pass would take about 36 minutes, and AutoGluon's memory pre-check refuses it unless its limit is raised. If you switch it on, it runs in its own process per feature set with `max_memory_usage_ratio = 8`, uses `predict_proba` with the development-chosen threshold (never AutoGluon's own tuned threshold), and scores a **declared, seeded sample** of development and validation pixels, stated beside its results. Expect about 20 extra minutes."""

CELL_SELECT = r"""# @title 3.1 Choose the foundation-model conditions
RUN_TABPFN3 = True  # @param {type:"boolean"}
RUN_TABICLV2 = True  # @param {type:"boolean"}
RUN_TABDPT = True  # @param {type:"boolean"}
RUN_MITRA = False  # @param {type:"boolean"}
MITRA_DEV_PIXELS = 10000  # @param {type:"integer"}
MITRA_VALIDATION_PIXELS = 10000  # @param {type:"integer"}
REQUIRE_ALL_SELECTED_MODELS = True
SELECTED_SYSTEMS = [s for s, on in (('C1', RUN_TABPFN3), ('C2', RUN_TABICLV2), ('C4', RUN_TABDPT), ('C3', RUN_MITRA)) if on]
if RUN_MITRA and not (1000 <= MITRA_DEV_PIXELS <= rt.DEV_SUBSET_ROWS and 1000 <= MITRA_VALIDATION_PIXELS):
    raise ValueError(f'MITRA_DEV_PIXELS must be 1,000-{rt.DEV_SUBSET_ROWS:,} and MITRA_VALIDATION_PIXELS at least 1,000')
tables = PREPARED['tables']['feature_sets']
display(pd.DataFrame([{
    'system': rt.FOUNDATION_SYSTEMS[s]['name'], 'context rows': rt.FM_FIT_ROWS,
    'features': ' / '.join(f"{fs} {tables[fs]['features']}" for fs in tables),
    'development pixels': MITRA_DEV_PIXELS if s == 'C3' else tables['s2m1']['dev']['rows'],
    'validation pixels': f'{MITRA_VALIDATION_PIXELS} (declared sample)' if s == 'C3' else 'all labelled',
    'prediction chunk rows': rt.FOUNDATION_SYSTEMS[s]['chunk_rows'], 'settings': rt.FOUNDATION_SYSTEMS[s]['settings'],
} for s in SELECTED_SYSTEMS]))"""

SEC3_ENV = r"""## 3.2 Isolated model environments

**Infrastructure — run the cell; reading its implementation is optional.** This cell is carried byte for byte from the FreshRetailNet v2 notebook in this repository: the model adapters at their pinned commits, the checkpoint manifests (revision and SHA-256 of every file), and one hash lock per model stack. A pinned `uv` (checked by size and SHA-256) creates one CPython 3.12.12 environment per selected model and installs only the locked wheels (`--require-hashes --only-binary :all: --no-deps`), then checks them with `uv pip check`. Environments are reused when their identity matches. Checkpoints are downloaded at their pinned revisions and verified against the manifest before use."""

CELL_ENV_TAIL = r"""
# --- rice notebook tail: build only the environments the selected systems need ---
SELECTED_ENVIRONMENTS = sorted({rt.FOUNDATION_SYSTEMS[s]['family'] for s in SELECTED_SYSTEMS})
MODEL_ENVIRONMENT_PYTHONS = {}
ENVIRONMENT_ERRORS = {}
for environment_key in SELECTED_ENVIRONMENTS:
    try:
        MODEL_ENVIRONMENT_PYTHONS[environment_key] = ensure_environment(environment_key)
    except Exception as exc:
        ENVIRONMENT_ERRORS[environment_key] = f"{type(exc).__name__}: {exc}"
        print(f"ERROR preparing {environment_key}: {ENVIRONMENT_ERRORS[environment_key]}")
if ENVIRONMENT_ERRORS and REQUIRE_ALL_SELECTED_MODELS:
    raise RuntimeError(f"Model environment(s) failed: {sorted(ENVIRONMENT_ERRORS)}; see Troubleshooting.")
ENVIRONMENT_FAILED_SYSTEMS = {s: ENVIRONMENT_ERRORS[rt.FOUNDATION_SYSTEMS[s]['family']] for s in SELECTED_SYSTEMS
                              if rt.FOUNDATION_SYSTEMS[s]['family'] in ENVIRONMENT_ERRORS}
SELECTED_SYSTEMS = [s for s in SELECTED_SYSTEMS if rt.FOUNDATION_SYSTEMS[s]['family'] in MODEL_ENVIRONMENT_PYTHONS]
MODEL_IDENTITIES = {
    key: {'model_id': CARRIED_MODEL_MANIFESTS[key]['modelId'], 'revision': CARRIED_MODEL_MANIFESTS[key]['revision'],
          'files': CARRIED_MODEL_MANIFESTS[key]['files'], 'adapter_revision': ADAPTER_REVISIONS[key],
          'environment_lock_sha256': ENVIRONMENT_LOCK_SHA256[key], 'python': FOUNDATION_PYTHON_SPEC}
    for key in MODEL_ENVIRONMENT_PYTHONS
}
display(pd.DataFrame([{'environment': k, **{f: v[f] for f in ('model_id', 'revision', 'adapter_revision')},
                       'lock': v['environment_lock_sha256'][:12]} for k, v in MODEL_IDENTITIES.items()]))"""

SEC3_RUN = r"""## 3.3 Run the foundation models on development

The next cell starts one child process per model (Mitra: one per feature set) with that model's environment. Each process loads the pinned checkpoint, registers the 10,000 fit rows as context, predicts the development subset in chunks of 8,192 rows, and saves its rice probabilities and the fitted artifact. The notebook then chooses each condition's threshold on development. Logs stream below; expect about 2–4 minutes per model."""

CELL_RUNNER = r"""# @title 3.3 Run the foundation models (development only)
def run_child(command, log_path, timeout_s=7200):
    # Streams the child's output; a hung process is killed after timeout_s and reported, never treated as success.
    print('$', ' '.join(str(c) for c in command), flush=True)
    process = subprocess.Popen([str(c) for c in command], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               bufsize=1, env=carried_environment())
    timer = threading.Timer(timeout_s, process.kill)
    timer.start()
    lines = []
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
            lines.append(line)
        code = process.wait()
    finally:
        timer.cancel()
    Path(log_path).write_text(''.join(lines), encoding='utf-8')
    if code != 0:
        raise RuntimeError(f'the model process exited with code {code}; its log is {log_path}')

def run_fm(system, mode, feature_sets, extra, out_dir):
    family = rt.FOUNDATION_SYSTEMS[system]['family']
    config = rt.runner_config(system, mode, feature_sets, extra, out_dir, weights_dir=MODEL_WEIGHTS_DIRS[family],
                              seed=RANDOM_SEED)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = f"{mode}-{'-'.join(feature_sets)}"
    config_path = Path(out_dir) / f'{stem}.config.json'
    config_path.write_text(json.dumps(config, indent=2), encoding='utf-8')
    run_child([MODEL_ENVIRONMENT_PYTHONS[family], FM_RUNNER, '--config', config_path], Path(out_dir) / f'{stem}.log')
    return json.loads(Path(config['result_path']).read_text(encoding='utf-8'))

FOUNDATION = rt.foundation_stage(LEDGER, SELECTED_SYSTEMS, run_fm, MODEL_IDENTITIES,
                                 mitra_dev_pixels=MITRA_DEV_PIXELS, require_all=REQUIRE_ALL_SELECTED_MODELS)
display(pd.DataFrame([{'condition': k, 'dev pixels': v['dev_pixels'], 'dev F1 at 0.5': v['dev_f1_at_0p5'],
                       'chosen threshold': v['threshold'], 'dev F1 at chosen': v['dev_f1_selected'],
                       **{f'{t} s': v['timings_s'][t] for t in ('load_s', 'fit_s', 'predict_dev_s')},
                       'peak GPU GiB': v['peak_gpu_gib'], 'device': v['device']}
                      for k, v in FOUNDATION['records'].items()]).round(4))
if FOUNDATION['errors']:
    display(pd.Series(FOUNDATION['errors'], name='error').to_frame())"""

AFTER_RUNNER = r"""**What to notice:** registering the context takes seconds; almost all the time goes to prediction, because every predicted row attends to the context. Peak GPU memory differs by model (TabICLv2 is the heaviest at 8,192-row chunks). The `device` column should read `cuda`."""

# --------------------------------------------------------------------------------------------------
# section 4: development comparison
# --------------------------------------------------------------------------------------------------

SEC4 = r"""# 4. Compare on development: the 2 × 5 grid [Evaluation practice]

The next cell lays out development F1 at each condition's chosen threshold as a grid: one row per system, one column per feature set. It also computes two numbers for your prediction: the **feature effect** (AlphaEarth minus M1, per system) and the **model spread** (best minus worst learned system, per feature set). The threshold curves show how sensitive each condition is to its threshold: a flat top means the choice is not fragile."""

CELL_DEV_GRID = r"""# @title 4.1 Development grid, feature effect, model spread and threshold curves
records = {**BASELINE, **FOUNDATION['records']}
grid = pd.DataFrame([{'system': rt.grid_name(v['name'], v.get('dev_pixels_rule', '')), 'feature_set': v['feature_set'],
                      'dev F1': v['dev_f1_selected']}
                     for v in records.values()]).pivot(index='system', columns='feature_set', values='dev F1')
display(grid.round(4))
learned = grid.drop(index=rt.BASELINE_SYSTEMS['A'], errors='ignore')
if {'alphaearth', 's2m1'} <= set(learned.columns):
    display(pd.DataFrame({'feature effect (AlphaEarth - M1)': learned['alphaearth'] - learned['s2m1']}).round(4))
    display(pd.Series({fs: learned[fs].max() - learned[fs].min() for fs in learned.columns},
                      name='model spread (best - worst learned system)').to_frame().round(4))
fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
for ax, fs in zip(axes, ('alphaearth', 's2m1')):
    for key, rec in records.items():
        if rec['feature_set'] != fs or rec['kind'] == 'trivial':
            continue
        folder = 'baseline' if rec['kind'] == 'lightgbm' else f"foundation/{rec['system']}/{fs}"
        name = f"{rec['system']}_{fs}_dev_proba.npy" if rec['kind'] == 'lightgbm' else 'dev_proba.npy'
        prob = np.load(LEDGER.root / folder / name).astype(float)
        _, dev_y = rt.load_split(LEDGER.root / 'tables', fs, 'dev')
        if rec['kind'] == 'foundation' and rec['indices']:
            dev_y = dev_y[np.load(rec['indices']['dev'])]
        sweep = rt.select_threshold([prob], [dev_y])['sweep']
        ax.plot([r['threshold'] for r in sweep], [r['f1'] for r in sweep], label=rec['name'])
    ax.set_title(rt.FEATURE_SET_LABELS[fs]); ax.set_xlabel('threshold'); ax.set_ylabel('development rice F1')
axes[0].legend(fontsize=7); plt.tight_layout(); plt.show()"""

AFTER_DEV_GRID = r"""**What to notice:** compare the size of the feature effect with the model spread. Then compare both with your prediction from Section 2.

**Checkpoint.** Is the gap between feature sets larger or smaller than the gap between models? Does any model change rank between the two feature sets?

<details><summary>Check your reasoning</summary>On these chips the feature-set gap is usually several times larger than the spread between learned models: every learned system gains a lot from AlphaEarth, and on a given feature set they land close together. That suggests the description of each pixel limits rice mapping here more than the classifier does. Small rank changes between models are within what different development subsets would shuffle; do not read them as a ranking. These are still development numbers.</details>"""

# --------------------------------------------------------------------------------------------------
# section 5: activity
# --------------------------------------------------------------------------------------------------

SEC5 = r"""# 5. Activity: change one thing — the context size [Evaluation practice]

**Question.** Is TabPFN-3's weaker result on M1 a data-volume problem (too few context rows) or a feature problem?

1. **Predict.** If TabPFN-3 sees 1,000, then 3,000, then 10,000 M1 fit rows, how much will development F1 change? Will 10,000 M1 rows reach the AlphaEarth result?
2. **Change one thing.** Only the number of context rows changes. The smaller contexts are nested inside the larger ones and keep the same rice share; everything else is fixed.
3. **Run** the next cell (about 1–2 minutes per size on a T4). You may change `ACTIVITY_CONTEXT_ROWS` (at most 10,000) or `ACTIVITY_FEATURE_SET` and re-run it before Section 6.
4. **Observe and explain** below.

The activity is development-only and changes nothing canonical: every model keeps the 10,000-row context in the frozen comparison. Its result is recorded in `selection.json` for interpretation."""

CELL_ACTIVITY = r"""# @title 5.1 TabPFN-3 context-size sweep on development
RUN_ACTIVITY = True  # @param {type:"boolean"}
ACTIVITY_FEATURE_SET = "s2m1"  # @param ["s2m1", "alphaearth"]
ACTIVITY_CONTEXT_ROWS = "1000, 3000, 10000"  # @param {type:"string"}
try:
    sizes = [int(part) for part in ACTIVITY_CONTEXT_ROWS.replace(',', ' ').split()]
except ValueError as exc:
    raise ValueError('ACTIVITY_CONTEXT_ROWS must be whole numbers separated by commas, e.g. "1000, 3000, 10000"') from exc
ACTIVITY = rt.activity_stage(LEDGER, run_fm, enabled=RUN_ACTIVITY, system='C1', feature_set=ACTIVITY_FEATURE_SET,
                             sizes=sizes, available=sorted({r['system'] for r in FOUNDATION['records'].values()}))
if ACTIVITY['ran']:
    table = pd.DataFrame(ACTIVITY['rows'])[['context_rows', 'context_rice', 'fit_s', 'predict_dev_s', 'dev_f1_at_0p5',
                                            'dev_selected_threshold', 'dev_f1_selected']]
    display(table.round(4))
    reference = FOUNDATION['records'].get('C1/alphaearth')
    if reference:
        print(f"For comparison, TabPFN-3 on AlphaEarth with 10,000 rows: development F1 {reference['dev_f1_selected']:.4f}")
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(table['context_rows'], table['dev_f1_selected'], marker='o')
    ax.set_xscale('log'); ax.set_xlabel('context rows'); ax.set_ylabel('development rice F1')
    ax.set_title(f'TabPFN-3 on {rt.FEATURE_SET_LABELS[ACTIVITY_FEATURE_SET]}'); plt.tight_layout(); plt.show()
else:
    print('Activity not run:', ACTIVITY['reason'])"""

AFTER_ACTIVITY = r"""**Observe and explain.** How much did F1 change from 1,000 to 10,000 rows, compared with the gap to the AlphaEarth row? What did the larger context cost in seconds?

<details><summary>Check your reasoning</summary>Typically F1 rises a little and then flattens as rows are added, while prediction time grows. If 10,000 M1 rows stay well below the AlphaEarth result, more rows of the same features do not close the gap: the limit is in what the features describe, not in how many examples the model sees. One caveat: development pixels come from a few regions, so a difference of a point or two between sizes is within noise. The 10,000-row setting should reproduce the canonical TabPFN-3 condition from Section 3.</details>"""

# --------------------------------------------------------------------------------------------------
# section 6-7: freeze, score
# --------------------------------------------------------------------------------------------------

SEC6 = r"""# 6. Freeze every choice [Evaluation practice]

Validation is for measuring, not choosing. The next cell writes `selection.json` — every condition's threshold, LightGBM's tree count, each model's identity (checkpoint revision, adapter commit, environment lock hash, settings) and the SHA-256 of every saved artifact file, the data manifest's hash, the activity's result and, if Mitra is on, its declared validation sample — then records its hash in `frozen.json`. The scoring cell refuses to run if anything changed. After scoring starts, no development cell can run again in this session: to change a choice, start a fresh runtime."""

CELL_FREEZE = r"""# @title 6.1 Freeze
declared = {'mitra_validation_pixels': MITRA_VALIDATION_PIXELS} if 'C3' in SELECTED_SYSTEMS else {}
SELECTION = rt.freeze_stage(LEDGER, **declared)
print('selection.json SHA-256:', LEDGER.receipt('freeze')['selection_sha256'])
display(pd.DataFrame([{'condition': k, 'threshold': v['threshold'], 'dev F1': v['dev_f1_selected']}
                      for k, v in SELECTION['conditions'].items()]).set_index('condition').round(4))"""

SEC7 = r"""# 7. Score validation once [Evaluation practice]

The next cell checks the freeze, then scores every frozen condition on **all 113,821 labelled validation pixels** (Mitra: its declared sample). Each foundation model is **reloaded from its saved files in a fresh process**, after checking every file's SHA-256 against the frozen inventory; before scoring, it re-predicts its first development chunk and must reproduce the saved probabilities within 1e-4. LightGBM is reloaded from its saved model file. Expect about 2–3 minutes per foundation model.

Each condition is reported at 0.5 and at its development-chosen threshold, pooled over all scored pixels."""

CELL_SCORE = r"""# @title 7.1 Score the frozen conditions on validation (once)
SCORED = rt.score_stage(LEDGER, run_fm)
METRICS = pd.read_csv(LEDGER.root / 'scores' / 'metrics.csv')
if SCORED['parity']:
    display(pd.DataFrame(SCORED['parity']).T[['rows', 'max_abs_difference', 'within_tolerance']])"""

SEC7_READ = r"""## 7.2 Read the results [Evaluation practice]

The grid below is validation rice F1 at each condition's frozen threshold. Then come the feature effect and model spread on validation, the full metric table, the validation class balance (shown only now), and a per-chip table for **diagnosis only**: pixels within a chip are strongly correlated and the chips cluster in two regions, so per-chip differences and small pooled differences are not significance tests [11]."""

CELL_RESULTS = r"""# @title 7.2 Validation grid, effects, full table and per-chip diagnosis
main = METRICS[(METRICS['split'] == 'validation') & (METRICS['threshold_rule'] == 'development-selected')]
main = main.assign(system=[rt.grid_name(s, p) for s, p in zip(main['system'], main['pixels'])])
GRID = main.pivot(index='system', columns='feature_set', values='f1')
display(GRID.round(4))
learned = GRID.drop(index=rt.BASELINE_SYSTEMS['A'], errors='ignore')
if {'alphaearth', 's2m1'} <= set(learned.columns):
    display(pd.DataFrame({'feature effect (AlphaEarth - M1)': learned['alphaearth'] - learned['s2m1']}).round(4))
    display(pd.Series({fs: learned[fs].max() - learned[fs].min() for fs in learned.columns},
                      name='model spread (best - worst learned system)').to_frame().round(4))
display(METRICS[['condition', 'split', 'pixels', 'threshold_rule', 'threshold', 'f1', 'iou', 'precision', 'recall',
                 'undefined']].round(4))
first = main.iloc[0]
print(f"Validation class balance (now that scoring is done): {int(first.tp + first.fn)} rice of "
      f"{int(first.tp + first.fp + first.fn + first.tn)} labelled pixels")
CHIP_TABLE = pd.read_csv(LEDGER.root / 'scores' / 'per_chip_metrics.csv')
sampled = set(main.loc[main['pixels'].str.startswith('declared'), 'condition'])
CHIP_TABLE['condition'] = [f'{c} (sample)' if c in sampled else c for c in CHIP_TABLE['condition']]
display(CHIP_TABLE.pivot(index='chip', columns='condition', values='f1').round(3))
ax = GRID.plot.bar(figsize=(7, 3.2), rot=20)
ax.set_ylabel('validation rice F1')
ax.set_title('Frozen conditions, all labelled validation pixels' + (' (Mitra: its declared sample)' if sampled else ''))
plt.tight_layout(); plt.show()"""

AFTER_RESULTS = r"""**What to notice:** blank F1 or recall in the per-chip table means it is undefined (a chip with no reference rice), not zero. Compare validation with development: a condition whose validation F1 drops far below its development F1 relied on something specific to the development regions. A's precision is undefined because it never predicts rice.

**Checkpoint.** (1) Does the validation grid tell the same story as the development grid? (2) Name one per-chip pattern that pooled F1 hides. (3) What would you need before claiming one foundation model is better than LightGBM for Philippine rice?

<details><summary>Check your reasoning</summary>(1) Usually yes: the feature effect stays much larger than the model spread. (2) For example, a chip where every system over-predicts rice (possibly label error or a crop that looks like rice), or the two rice-free chips where only false positives can occur. (3) More validation chips from more regions and years, scored by whole region, with spatially aware uncertainty [11]; a difference of a point or two of F1 on 8 chips from two regions does not establish a ranking.</details>"""

# --------------------------------------------------------------------------------------------------
# section 8: reload + export
# --------------------------------------------------------------------------------------------------

SEC8 = r"""# 8. Reload check and export [Engineering]

**Infrastructure — run the cell; reading its implementation is optional.** One frozen foundation-model condition (TabPFN-3 on AlphaEarth by default) is reloaded once more in a fresh process and re-predicts the first 8,192 validation pixels; the probabilities must match the scored ones within 1e-4. Loading a model is not the same as reproducing it: this check compares outputs, not only that the files open.

`results.zip` then collects `run_summary.json`, `data_manifest.json` (INDEX digest, chip digests, roles), `features.json` (recipes and fills), `selection.json` and `frozen.json`, the development and validation metric tables, `per_chip_metrics.csv`, `activity.csv`, `verification.json`, per-pixel validation predictions (`validation_predictions.csv.gz`: chip, pixel, label and each condition's rice probability), the LightGBM model files, the stage receipts, attribution and a conclusion template, with `checksums.json`. The foundation-model artifacts stay in this runtime (their file inventories are in `selection.json`): they embed the fit rows as context, and the bundle carries no model weights and no imagery. The TabICLv2 artifact is a pickle, which runs code when loaded: only load pickles you created yourself."""

CONCLUSION_TEMPLATE = r"""# Conclusion (fill in from your run)

- Task: rice / non-rice per 20 m pixel on 8 validation chips (2 regions), scored once after a frozen protocol.
- Feature effect: AlphaEarth minus M1 validation F1 for each learned system: ...
- Model spread: best minus worst learned system on each feature set: ...
- Baseline: LightGBM (65,536 rows) F1 ... ; all-non-rice F1 0.
- Activity: TabPFN-3 context 1,000 -> 10,000 rows changed development F1 by ... at a cost of ... s.
- Largest per-chip failure and its likely cause: ...
- Limits: 8 chips, 2 regions, spatially correlated pixels, labels from a published map, possible AlphaEarth pretraining overlap; tutorial evidence, not a model ranking.
"""

CELL_EXPORT = r"""# @title 8.1 Reload one condition in a fresh process, then export results.zip
PROBE = rt.probe_reload(LEDGER, run_fm)
display(pd.Series(PROBE, name='validation reload check').to_frame())
receipts = {p.stem: json.loads(p.read_text()) for p in sorted(LEDGER.receipts.glob('*.json'))}
RUN_SUMMARY = {
    'notebook': 'tutorials/DIMER_Philippines_Rice_Pixel_Tables_Workshop.ipynb', 'workshop_revision': WORKSHOP_REVISION,
    'repository': 'kurtvalcorza/mitra-classifier-pipeline', 'notebook_spec': '2.4', 'profile': 'E2E', 'mode': 'WORKSHOP',
    'carried_sha256': CARRIED_HASHES, 'adapter_sha256': CARRIED_SOURCE_SHA256, 'environment_lock_sha256': ENVIRONMENT_LOCK_SHA256,
    'host': HOST, 'model_identities': MODEL_IDENTITIES, 'systems': ['A', 'B', 'B10k', *sorted({r['system'] for r in FOUNDATION['records'].values()})],
    'failed_systems': sorted({*FOUNDATION['errors'], *ENVIRONMENT_FAILED_SYSTEMS}),
    'environment_errors': ENVIRONMENT_ERRORS,
    'selection_sha256': receipts['freeze']['selection_sha256'], 'stage_receipts': receipts, 'reload_probe': PROBE,
    'wall_seconds': round(time.perf_counter() - SESSION_STARTED, 1),
    'evidence_level': 'tutorial evidence on 8 validation chips from 2 regions; not a benchmark or model ranking',
}
EXPORT = rt.export_stage(LEDGER, RUN_SUMMARY, CONCLUSION_TEMPLATE)
print(f"results.zip: {EXPORT['members']} files, checksums verified, SHA-256 {EXPORT['sha256']}")
display(FileLink(str(EXPORT['path'])))"""

SEC9 = r"""# 9. Conclude with evidence [Evaluation practice]

Write your conclusion in five sentences, using only numbers this run printed (the template is also in `results.zip`):

1. **Task and evidence:** what was predicted, on how many validation chips and regions, scored how many times.
2. **Principal result:** the feature effect and the model spread on validation, with the numbers.
3. **Baseline:** how the foundation models compare with LightGBM on the same feature set, and with B10k on the same rows.
4. **Failure mode or uncertainty:** the largest per-chip failure and a plausible cause (model, label, cloud gaps, or region).
5. **Limits:** what 8 clustered chips, published-map labels and possible AlphaEarth pretraining overlap prevent you from claiming.

**Transfer.** The same protocol applies to any per-pixel mapping task with chip-level labels: change the label (for example, mangrove or sugarcane), keep the region-level roles and the freeze, and check whether the feature effect still dominates.

## What did not work

These design choices were tried in development runs on fresh Colab T4 runtimes (2026-10-11; development evidence, not results of this run) and rejected:

| Tried | Observed | Decision |
|---|---|---|
| Mitra through AutoGluon 1.5.0 with the v2 notebook's `max_memory_usage_ratio = 1.10` | AutoGluon refused Mitra before loading it: its pre-check estimated about 45 GB for 10,000 × 65 rows against about 10 GB free; the real peak was under 3.2 GB of RAM | Mitra runs only on request, in its own process, with the ratio raised to 8 |
| Mitra scoring every development and validation pixel | About 50 rows per second on a T4 at both 4,096 and 10,000 context rows: about 2.6 hours for full scoring | Opt-in, on a declared seeded sample of 10,000 pixels |
| AutoGluon's own `predict()` threshold for Mitra | AutoGluon tuned it to about 0.06–0.33 on its tuning rows, a different rule from every other system | `predict_proba` with the notebook's development-chosen threshold |
| Choosing thresholds on all 119,885 development pixels | Roughly doubled foundation-model prediction time for a threshold that moves little | A seeded 30,000-pixel subset that keeps the development rice share |"""

# --------------------------------------------------------------------------------------------------
# section 10: BYOD
# --------------------------------------------------------------------------------------------------

SEC10 = r"""# 10. Optional: bring your own chips [Engineering]

This section re-runs the **whole protocol** on your chips — validate, split by site, stage tables, fit every selected system, choose thresholds on your development chips, freeze, score your test chips once and export `byod_results.zip`. It is off by default and is not part of Run all.

**Schema (checked before any model runs).** One HDF5 file per chip-year (`<site>_<year>.h5`), with:

| Dataset | Shape | Type and values |
|---|---|---|
| `S2` | 24 × 6 × 128 × 128 | float surface reflectance (0–1 scale, up to 2.0); bins and band order as in the data context |
| `S2_valid` | 24 × 128 × 128 | 0 or 1 |
| `label` | 128 × 128 | 0 non-rice, 1 rice, 255 ignore |
| `AlphaEarth` + `AE_valid` (optional, together) | 64 × 128 × 128 and 128 × 128 | float embedding; 0 or 1 |

A site is the HDF5 attribute `group` or `group_id`, else the file name without its `_YYYY` year. **Minimum:** 6 labelled chips from at least 3 sites, with rice in fit, development and test sites and non-rice in fit. If any labelled chip lacks AlphaEarth, only the M1 column runs. Chips whose label is all 255 are reported as prediction-only and not used. **Limits:** at most 200 chips and 4 GiB; zip archives with absolute paths, `..`, backslashes or symbolic links are refused. A refusal names the file and the rule it broke.

**Your data stays in this runtime;** nothing is uploaded elsewhere, and only files you download leave it. **Do not upload confidential, restricted, personal or regulated data** unless you are authorised to process it in this runtime. The fitted foundation-model artifacts contain your fit rows as context.

Set `USE_BYOD = True`, then either give `BYOD_PATH` (folders, `.zip` or `.h5` paths, comma-separated) or leave it blank to open the Colab upload dialog. The foundation models are the ones selected in Section 3.1."""

CELL_BYOD = r"""# @title 10.1 Bring your own chips (re-adapts, freezes and scores your test chips once)
USE_BYOD = False  # @param {type:"boolean"}
BYOD_PATH = ""  # @param {type:"string"}
if USE_BYOD:
    BYOD_ROOT = RUN_ROOT / 'byod'
    if BYOD_ROOT.exists():  # a new attempt never inherits an earlier attempt's chips, freeze or scores
        shutil.rmtree(BYOD_ROOT)
    sources = [Path(p.strip()).expanduser() for p in BYOD_PATH.split(',') if p.strip()]
    if not sources:
        from google.colab import files  # only on Colab, only after the sample path has completed
        upload_dir = SESSION_ROOT / 'byod_upload'
        if upload_dir.exists():  # a retry never mixes in files from an earlier upload
            shutil.rmtree(upload_dir)
        upload_dir.mkdir(parents=True)
        for name, payload in files.upload().items():
            (upload_dir / Path(name).name).write_bytes(payload)
        sources = [upload_dir]
    missing_sources = [str(s) for s in sources if not s.exists()]
    if missing_sources:
        raise FileNotFoundError(f'BYOD_PATH entries not found: {missing_sources}')
    paths = rt.collect_byod(sources, BYOD_ROOT / 'input')
    if not paths:
        raise rt.ChipError('no .h5 chips found; see the schema above')
    byod_chips, problems = {}, []
    for path in paths:
        try:
            byod_chips[path.stem] = rt.read_chip_h5(path, require_label=False)
        except rt.ChipError as exc:
            problems.append(str(exc))
    if problems:
        raise rt.ChipError('BYOD validation refused the upload:\n  ' + '\n  '.join(problems))
    plan = rt.byod_plan(byod_chips)
    print(f"{len(paths)} chips valid; adapting on {len(plan['labelled'])} labelled chips; feature sets: {plan['feature_sets']}")
    BYOD = rt.Ledger(BYOD_ROOT)
    byod_prepared = rt.prepare_byod(BYOD, byod_chips, plan, {p.stem: rt.sha256_file(p) for p in paths})
    del byod_chips; gc.collect()
    display(pd.DataFrame(byod_prepared['chips']))
    rt.baseline_stage(BYOD)
    rt.foundation_stage(BYOD, SELECTED_SYSTEMS, run_fm, MODEL_IDENTITIES, mitra_dev_pixels=MITRA_DEV_PIXELS,
                        require_all=REQUIRE_ALL_SELECTED_MODELS)
    rt.activity_stage(BYOD, run_fm, enabled=False, system='C1', feature_set='s2m1', sizes=[], available=[])
    rt.freeze_stage(BYOD, **({'mitra_validation_pixels': MITRA_VALIDATION_PIXELS} if 'C3' in SELECTED_SYSTEMS else {}))
    rt.score_stage(BYOD, run_fm, split_label='test')
    byod_metrics = pd.read_csv(BYOD_ROOT / 'scores' / 'metrics.csv')
    byod_main = byod_metrics[(byod_metrics['split'] == 'test') & (byod_metrics['threshold_rule'] == 'development-selected')]
    byod_main = byod_main.assign(system=[rt.grid_name(s, p) for s, p in zip(byod_main['system'], byod_main['pixels'])])
    display(byod_main.pivot(index='system', columns='feature_set', values='f1').round(4))
    byod_export = rt.export_stage(BYOD, {'byod': True, 'plan': {k: plan[k] for k in ('split', 'feature_sets', 'prediction_only')},
                                         'model_identities': MODEL_IDENTITIES, 'host': HOST}, CONCLUSION_TEMPLATE)
    byod_zip = BYOD_ROOT / 'byod_results.zip'
    shutil.copy2(byod_export['path'], byod_zip)
    print(f"byod_results.zip: {byod_export['members']} files, verified; it stays in this runtime until you download it")
    display(FileLink(str(byod_zip)))
else:
    print('BYOD not run (USE_BYOD = False).')"""

CELL_SUMMARY = r"""# @title Run-all completion summary
completion = {'profile': 'E2E', 'mode': 'WORKSHOP', 'stages finished': sorted(p.stem for p in LEDGER.receipts.glob('*.json')),
              'conditions scored': len(SELECTION['conditions']), 'reload probe': PROBE.get('within_tolerance', 'not run'),
              'results.zip': str(EXPORT['path']), 'wall minutes': round((time.perf_counter() - SESSION_STARTED) / 60, 1)}
display(pd.Series(completion, name='value').to_frame())
expected = {'prepare', 'baseline', 'foundation', 'activity', 'freeze', 'score', 'export'}
if not expected <= set(completion['stages finished']):
    raise RuntimeError(f"Run all did not finish every stage: missing {sorted(expected - set(completion['stages finished']))}")
print('Run all complete: data verified, baselines and foundation models developed, choices frozen, validation scored once, '
      'artifact reload checked, results exported.')"""

TROUBLE = r"""# Troubleshooting

| What you see | Likely cause | What to do |
|---|---|---|
| "needs a Linux x86_64 runtime" | Windows, macOS or ARM kernel | Use Google Colab, Kaggle or a Linux x86_64 Jupyter server |
| "No GPU detected" or `device` = `cpu` | CPU runtime | Runtime → Change runtime type → T4 GPU, then Run all |
| `INDEX.jsonl SHA-256 ... != pinned` or a chip SHA-256 differs | The download changed or is corrupt | Re-run once; if it persists, stop and report it. Do not remove the check |
| Kaggle download fails | Network or Kaggle outage; no account is needed | Retry once later |
| uv, wheel or checkpoint hash mismatch | A pinned download changed or was truncated | Re-run once; a persistent mismatch must be reported, not bypassed |
| Out of disk | Each environment is 5–7.5 GB | Start a fresh runtime with at least 30 GB free; keep Mitra off |
| CUDA out of memory | Another process holds GPU memory, or a non-T4 GPU with less memory | Restart the runtime and Run all; never record a skipped model as a success |
| Mitra: "Estimated to require ... GB" | AutoGluon's memory pre-check | Keep `max_memory_usage_ratio = 8` (set by the notebook) and run Mitra on a fresh runtime |
| "needs the ... stage(s) to finish first" | A cell was run out of order | Runtime → Run all from the top |
| "already scored ... exactly once" | A development or scoring cell was re-run after scoring started | Start a fresh runtime (Runtime → Disconnect and delete runtime) to change any choice |
| "changed after the freeze" or "differs from the inventory" | `selection.json` or a saved artifact was edited | Start a fresh runtime; scores from an edited freeze are not valid |
| "reload parity failed" | A reloaded model did not reproduce its probabilities | Keep `verification.json`, `probe.json` and the logs, and report it |
| A model process exits with a code | See the log path in the message | Read the last lines of the log; most causes are memory or download errors above |
| BYOD refused | The message names the file and the rule | Fix that file (shape, value range, labels, AlphaEarth pairing, sites) and re-run the BYOD cell |"""

LICENCES = r"""# Data, models, attribution and licences

**Data:** `kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching`, version 1 [10], CC BY 4.0, verified by `INDEX.jsonl` SHA-256 `e92bbb9f20d1f7efb225d568e2a44a6ae3d59faab9079590e2dfad780458344d` and every chip's SHA-256. Component attribution:

- **Rice labels:** Chen et al., 20 m paddy rice maps for Monsoon Asia 2018–2021 [9], CC BY 4.0 — reprojected, binarised and given an ignored edge band.
- **Optical inputs:** contains modified Copernicus Sentinel data 2019–2021 [8], under the Copernicus Sentinel data terms.
- **AlphaEarth Foundations Satellite Embedding** (Google and Google DeepMind) [7], CC BY 4.0.
- **Regional boundaries:** geoBoundaries PHL ADM1, CC BY 3.0 IGO.
- **Dataset compilation:** Kurt Valcorza, CC BY 4.0. Component terms continue to apply.

**Models** (each verified against its carried manifest before use):

| System | Checkpoint and revision | Licence | Adapter |
|---|---|---|---|
| TabPFN-3 | `Prior-Labs/tabpfn_3` @ `24a16a89d245878b846555110985634aa2e656d7` | **TabPFN-3 licence v1.0 — non-commercial use only** | `kurtvalcorza/tabpfn-classifier-pipeline` @ `769fb7b4` |
| TabICLv2 | `jingang/TabICL` @ `4dcd344ece2c00be9e831fdd35bed57b5ad83e19` | BSD-3-Clause | `kurtvalcorza/tabicl-classifier-pipeline` @ `7400516a` |
| TabDPT v1.2 | `Layer6/TabDPT` @ `4462ffbd1d8dea25d4862d30beed4b70cd596ae5` | Apache-2.0 | `kurtvalcorza/tabdpt-classifier-pipeline` @ `1e35dc47` |
| Mitra (opt-in) | `autogluon/mitra-classifier` @ `c425e9fa0910a6be1c494321792e7ba2a1367b1a` | Apache-2.0 | `kurtvalcorza/mitra-classifier-pipeline` @ `469d9125` |
| LightGBM | the runtime's `lightgbm` package | MIT | — |

The TabPFN-3 conditions this notebook fits, and their outputs, are for teaching and research only.

**AI Assistance Disclosure:** Generative AI assisted this notebook's code and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results and making release decisions. AI assistance is not independent verification, provider endorsement or release approval.

## References

[1] N. Hollmann, S. Müller, L. Purucker, A. Krishnakumar, M. Körfer, S. B. Hoo, R. T. Schirrmeister, and F. Hutter, "Accurate predictions on small data with a tabular foundation model," *Nature*, vol. 637, pp. 319–326, 2025, doi: 10.1038/s41586-024-08328-6. [Online]. Available: https://doi.org/10.1038/s41586-024-08328-6

[2] Prior Labs, "TabPFN-3 (tabpfn_3) model card," Hugging Face, revision 24a16a89d245878b846555110985634aa2e656d7. [Online]. Available: https://huggingface.co/Prior-Labs/tabpfn_3/tree/24a16a89d245878b846555110985634aa2e656d7

[3] J. Qu, D. Holzmüller, G. Varoquaux, and M. Le Morvan, "TabICL: A tabular foundation model for in-context learning on large data," arXiv:2502.05564, 2025, doi: 10.48550/arXiv.2502.05564. Model card at the pinned revision: https://huggingface.co/jingang/TabICL/tree/4dcd344ece2c00be9e831fdd35bed57b5ad83e19. [Online]. Available: https://doi.org/10.48550/arXiv.2502.05564

[4] J. Ma, V. Thomas, R. Hosseinzadeh, A. Labach, H. Kamkari, J. C. Cresswell, K. Golestan, G. Yu, A. L. Caterini, and M. Volkovs, "TabDPT: Scaling tabular foundation models on real data," arXiv:2410.18164, 2024, doi: 10.48550/arXiv.2410.18164. [Online]. Available: https://doi.org/10.48550/arXiv.2410.18164

[5] AutoGluon, "Mitra classifier model card," Hugging Face, revision c425e9fa0910a6be1c494321792e7ba2a1367b1a. [Online]. Available: https://huggingface.co/autogluon/mitra-classifier/tree/c425e9fa0910a6be1c494321792e7ba2a1367b1a

[6] G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A highly efficient gradient boosting decision tree," in *Proc. Adv. Neural Inf. Process. Syst. (NeurIPS)*, 2017, pp. 3146–3154. [Online]. Available: https://papers.nips.cc/paper/6907-lightgbm-a-highly-efficient-gradient-boosting-decision-tree

[7] C. F. Brown et al., "AlphaEarth Foundations: An embedding field model for accurate and efficient global mapping from sparse label data," arXiv:2507.22291, 2025, doi: 10.48550/arXiv.2507.22291. [Online]. Available: https://doi.org/10.48550/arXiv.2507.22291

[8] M. Drusch et al., "Sentinel-2: ESA's optical high-resolution mission for GMES operational services," *Remote Sens. Environ.*, vol. 120, pp. 25–36, 2012, doi: 10.1016/j.rse.2011.11.026. [Online]. Available: https://doi.org/10.1016/j.rse.2011.11.026

[9] Y. Chen, S. Liang, W. Li, H. Ma, H. Fang, and J. Xu, "20 m resolution maps of paddy rice cropping intensity and planting dates in Monsoon Asia from 2018 to 2021," Zenodo, 2025, doi: 10.5281/zenodo.15124936. [Online]. Available: https://doi.org/10.5281/zenodo.15124936

[10] K. Valcorza, "rice-philippines-s2-alphaearth-v1-teaching," Kaggle dataset, version 1, CC BY 4.0. [Online]. Available: https://www.kaggle.com/datasets/kurtvalcorza/rice-philippines-s2-alphaearth-v1-teaching/versions/1

[11] D. R. Roberts et al., "Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure," *Ecography*, vol. 40, no. 8, pp. 913–929, 2017, doi: 10.1111/ecog.02881. [Online]. Available: https://doi.org/10.1111/ecog.02881

[12] T. Saito and M. Rehmsmeier, "The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets," *PLOS ONE*, vol. 10, no. 3, e0118432, 2015, doi: 10.1371/journal.pone.0118432. [Online]. Available: https://doi.org/10.1371/journal.pone.0118432

**Related notebooks and sources:** the Prithvi rice capstone (`kurtvalcorza/prithvi-eo-feature-extraction-pipeline`, `tutorials/DIMER_Philippines_Rice_Mapping_Capstone.ipynb`) scores the same chips with a satellite foundation model; the FreshRetailNet v2 notebook in this repository supplies the carried adapters and locks; the specification is `docs/RICE_TABULAR_WORKSHOP_SPEC.md`."""

INFRA = {"cellView": "form", "jupyter": {"source_hidden": True}}


def cells() -> list[tuple[str, str, dict]]:
    files = carried_files()
    return [
        ("markdown", TITLE, {}), ("markdown", DATA_CONTEXT, {}),
        ("markdown", SEC0, {}), ("code", cell_setup(files), INFRA),
        ("markdown", SEC1, {}), ("code", CELL_PREPARE, {}), ("markdown", AFTER_PREPARE, {}),
        ("markdown", SEC1_LOOK, {}), ("code", CELL_LOOK, {}), ("markdown", AFTER_LOOK, {}),
        ("markdown", SEC2, {}), ("code", CELL_BASELINE, {}), ("markdown", AFTER_BASELINE, {}),
        ("markdown", SEC3, {}), ("code", CELL_SELECT, {}),
        ("markdown", SEC3_ENV, {}), ("code", v2_environment_head() + CELL_ENV_TAIL, INFRA),
        ("markdown", SEC3_RUN, {}), ("code", CELL_RUNNER, {}), ("markdown", AFTER_RUNNER, {}),
        ("markdown", SEC4, {}), ("code", CELL_DEV_GRID, {}), ("markdown", AFTER_DEV_GRID, {}),
        ("markdown", SEC5, {}), ("code", CELL_ACTIVITY, {}), ("markdown", AFTER_ACTIVITY, {}),
        ("markdown", SEC6, {}), ("code", CELL_FREEZE, {}),
        ("markdown", SEC7, {}), ("code", CELL_SCORE, {}),
        ("markdown", SEC7_READ, {}), ("code", CELL_RESULTS, {}), ("markdown", AFTER_RESULTS, {}),
        ("markdown", SEC8, {}), ("code", "CONCLUSION_TEMPLATE = " + repr(CONCLUSION_TEMPLATE) + "\n" + CELL_EXPORT, INFRA),
        ("markdown", SEC9, {}),
        ("markdown", SEC10, {}), ("code", CELL_BYOD, {}),
        ("code", CELL_SUMMARY, {}),
        ("markdown", TROUBLE, {}), ("markdown", LICENCES, {}),
    ]


def build() -> str:
    files = carried_files()
    nb_cells = []
    for index, (kind, source, metadata) in enumerate(cells()):
        cell = {"cell_type": kind, "id": f"rice-tables-{index:02d}", "metadata": metadata,
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb_cells.append(cell)
    notebook = {
        "nbformat": 4, "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"}, "accelerator": "GPU",
            "colab": {"name": NOTEBOOK.name, "gpuType": "T4", "provenance": []},
            "workshop_revision": WORKSHOP_REVISION,
            "dimer": {
                "notebook_spec": NOTEBOOK_SPEC, "notebook_profile": "E2E", "notebook_mode": "WORKSHOP",
                "notebook_kind": "WORKSHOP", "standalone": True, "requires_dimer_worker": False,
                "release_status": "Candidate", "clean_runtime_evidence": "pending",
                "specification": "docs/RICE_TABULAR_WORKSHOP_SPEC.md",
                "owner_decisions": {
                    "D1": "E2E / WORKSHOP", "D2": "Mitra opt-in, off by default, declared seeded samples",
                    "D3": "10,000 class-stratified fit rows for every foundation model",
                    "D4": "seeded 30,000-pixel development subset; validation scored in full",
                    "D5": "BYOD of schema-matched HDF5 with re-adaptation", "D6": "MOMENT out of scope for this revision",
                    "D7": "TabPFN-3 included with the non-commercial licence statement", "D8": "TabPFN-3 context-size activity",
                },
                "generated_from": {
                    "repository": "kurtvalcorza/mitra-classifier-pipeline", "generator": GENERATOR,
                    "files": {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in files.items()},
                    "v2_environment_head_sha256": hashlib.sha256(v2_environment_head().encode("utf-8")).hexdigest(),
                },
            },
        },
        "cells": nb_cells,
    }
    return json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="fail if the committed notebook is not up to date")
    args = parser.parse_args(argv)
    text = build()
    if args.check:
        current = NOTEBOOK.read_bytes().decode("utf-8") if NOTEBOOK.exists() else ""
        if current != text:
            print(f"STALE: {NOTEBOOK} differs from the generator output; run python {GENERATOR}", file=sys.stderr)
            return 1
        print(f"OK: {NOTEBOOK.name} is up to date ({len(text.encode('utf-8'))} bytes)")
        return 0
    NOTEBOOK.write_bytes(text.encode("utf-8"))
    print(f"wrote {NOTEBOOK} ({len(text.encode('utf-8'))} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
