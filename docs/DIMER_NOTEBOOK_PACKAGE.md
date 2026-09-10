# DIMER notebook model-package contract

This document separates two different artifact boundaries that must not be conflated.

## Current DIMER platform checkpoint boundary

The pipeline's current DIMER deployment contract accepts an uploaded checkpoint directory
containing exactly the model files needed by Mitra at load time: `model.safetensors` and
`config.json`, mounted through `DIMER_MODEL_DIR`. DIMER is **not** claimed here to emit a
self-describing ZIP with a notebook manifest.

## Repository-defined notebook transport envelope

DIMER Notebook Specification v1.0 requires the standalone tutorial's offline package path to
verify model identity, immutable revision, expected files, and digests before load. This repository
therefore defines a small transport envelope for notebook use. It is built from the DIMER checkpoint
pair by `scripts/build_dimer_notebook_package.py` and contains exactly:

```text
dimer-model-manifest.json
model.safetensors
config.json
```

The manifest schema is:

```json
{
  "manifest_version": "1.0",
  "model_id": "autogluon/mitra-classifier",
  "revision": "c425e9fa0910a6be1c494321792e7ba2a1367b1a",
  "files": {
    "model.safetensors": "<sha256>",
    "config.json": "<sha256>"
  }
}
```

For this release the producer refuses to package checkpoint bytes unless their SHA-256 values equal
the pinned public Mitra release digests already enforced by the tutorial.

## Build the notebook package

Given a DIMER checkpoint directory containing the two files:

```bash
python scripts/build_dimer_notebook_package.py \
  --checkpoint-dir /path/to/checkpoint \
  --output mitra-dimer-notebook-package.zip
```

The script prints the resulting ZIP SHA-256. The tutorial's `MODEL_SOURCE = 'DIMER notebook package'`
path then performs its own independent archive, manifest, model/revision, file-set, resource-ceiling,
and payload-digest checks before staging either model file.

## Evidence and scope

The producer has a standard-library self-test exercised by repository CI. That test proves the
repository-defined wrapper can emit the documented three-member envelope and correct manifest for
known input bytes. It does **not** prove DIMER platform distribution of this ZIP; the native DIMER
checkpoint boundary remains the two-file directory described above and in `DEPLOYMENT.md`.
