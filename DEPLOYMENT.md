# Deployment and operations — Mitra Classifier pipeline

This runbook is for the DIMER platform administrator and the AI engineer who connects the
pipeline. It lists what an administrator provisions and checks to deploy and operate the
pipeline. It complements the [README](README.md) (pipeline reference) and the platform's own
pipeline setup guide.

## Prerequisites

The administrator completes these before the pipeline can build or run.

1. **Portal access.** Grant the pipeline owner the AI Engineer role. The `tester` role cannot
   open the developers portal.
2. **GitHub App.** Install the DIMER GitHub App on the account that holds the repositories
   (`kurtvalcorza`), so the portal can read:
   - `mitra-classifier-dataset-validator`
   - `mitra-classifier-finetuner`

   A repository outside the app's default organization is invisible to the portal until the
   app is installed on its account. The portal reports this only when a build fails.
3. **Resource profile.** Provision a GPU and memory profile larger than the default — see
   [Resource profile](#resource-profile). The default `1 GPU, 8Gi` is too small for Mitra.
4. **Network egress.** If the fine-tuner image does not bake the weights, allow egress to
   `huggingface.co` — see [Weights delivery](#weights-delivery).
5. **Taxonomy.** None required. The pipeline uses the **Custom / Other** task type and a
   free-text base model, so no new Category, Task, or Subtask row is needed.

## Resource profile

Each fine-tuning run is a Kubernetes job. Size its profile to the deployment:

| Instance | GPU | Memory | Notes |
|---|---|---|---|
| GPU (default) | 1 | 12 Gi minimum, 16 Gi for large tables | Fine-tune mode. Mitra used ~10 GB on ~4,200 rows; AutoGluon skips the model if the projected footprint exceeds ~90% of available memory |
| CPU-only | 0 | 12 Gi minimum | Zero-shot mode (fine-tuning needs a GPU). The default GPU image auto-falls-back to CPU, or use the lean `Dockerfile.cpu` |

The validator is CPU-only and runs under a small profile.

If memory is under-provisioned, AutoGluon trains no model and the run fails with
`No models were trained successfully during fit()`. This is a resource signal, not a data
error: raise the memory profile and re-run.

## Weights delivery

The fine-tuner needs the Mitra weights (`autogluon/mitra-classifier`). Choose one:

| Option | Egress | Reproducibility | How |
|---|---|---|---|
| A — bake into image (recommended) | none at runtime | deterministic | Uncomment Option A in the fine-tuner `Dockerfile`; it downloads the pinned revision at build time |
| B — runtime fetch | requires `huggingface.co` | checksum-verified | Default; AutoGluon downloads on first run |
| C — uploaded weights | none at runtime | uploaded bytes | Mount a checkpoint dir (`model.safetensors` + `config.json`) and set `DIMER_MODEL_DIR`; the fine-tuner installs and uses those exact bytes |

Pinned revision: `c425e9fa0910a6be1c494321792e7ba2a1367b1a`, `model.safetensors` SHA-256
`e06a055e…`, `config.json` SHA-256 `2c96c24d…`. AutoGluon's Mitra loader takes a repo id and no
revision argument, so the base weights cannot be pinned by revision through it. The fine-tuner
instead **verifies both the loaded `model.safetensors` and `config.json` SHA-256s against the
expected values before fitting and fails the run on a mismatch** (options A/B); `result.json`'s
`provenance` block records the resolved revision, the loaded `weightsSha256`/`configSha256`, the
source, and whether the check was enforced. Option C records the uploaded bytes' checksums but
does not compare them to the public pinned values.

## Build and enable

1. Create the pipeline in **AI Engineer → New Pipeline** with the values in the README
   [Creating the pipeline](README.md#creating-the-pipeline) section.
2. Build the validator image.
3. Build the fine-tuner image. The fine-tuner build re-reads `dimer-pipeline.json`.
4. Run the smoke test with a small dataset. The dataset must pass validation before
   fine-tuning unlocks.
5. Enable the pipeline once the enable checklist is complete.

There is no push-to-rebuild webhook. After a code change, rebuild the affected image
manually — the validator card for a validator change, the fine-tuner card for a training or
`dimer-pipeline.json` change.

## Container image and CI

The fine-tuner image is built **`FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`** — CUDA
12.8 / torch 2.8 ships `sm_120` (Blackwell, e.g. RTX 5070 Ti) kernels and satisfies AutoGluon
1.5.0's `torch>=2.6,<2.10`, so pip does not swap the base torch. The build asserts the installed
`torch`/`CUDA` after `pip install` and fails on drift. Do not downgrade to `cuda12.4`/`torch2.5`:
that lacks `sm_120` and dies with "no kernel image is available" on RTX 50-series / B200 GPUs.

Two GitHub Actions workflows guard the repo:

- **`ci`** (every push/PR) — compiles the deployable sources, runs the unit suite, and enforces
  the shared dataset-resolution block is byte-identical across the validator/finetuner copies and
  matches the cross-repo pinned SHA (so this repo and the standalone deployment repos cannot
  drift). It deliberately does **not** install AutoGluon.
- **`integration`** (manual `workflow_dispatch`; nightly only when the repo variable
  `ENABLE_NIGHTLY_GPU` is `true`) — needs a **self-hosted runner labelled `gpu`** with the NVIDIA
  Container Toolkit. It builds the image, loads Mitra offline from the pinned weights, fine-tunes
  a tiny model, saves it, reloads the `TabularPredictor`, and predicts — catching torch/base-image
  drift, Mitra API changes, offline-weight failures, and seed/metric propagation that the unit
  suite cannot. Attach a GPU runner and set `ENABLE_NIGHTLY_GPU` to turn on the nightly run.

## Operations

- **Fine-tune versus zero-shot.** Fine-tuning Mitra requires a GPU. On a GPU node the run
  fine-tunes Mitra's weights; on a CPU node (or the GPU image run without a GPU) it runs Mitra
  **zero-shot** — in-context inference with no weight update — automatically. Each run records
  the effective `mode` (`fine-tune`/`zero-shot`) and `device` in `result.json`. Zero-shot is
  faster and CPU-safe, at some cost in accuracy.
- **Binary versus multiclass.** The fine-tuner infers the problem type from the number of
  distinct target values (2 → binary, 3–10 → multiclass) and records it in `result.json`
  (`metrics.problemType`, `metrics.numClasses`). A target with more than 10 classes fails the
  run with a clear message; reduce the class count in the dataset.
- **Monitoring.** Each run writes `result.json` with metrics and a `provenance` block
  (base-model revision, dataset SHA-256, AutoGluon version). For a failed run, open the
  platform's diagnostics terminal for job status, pod exit codes, Kubernetes events, and the
  container log tail.
- **Auditing.** The `provenance` block plus the container image tag form the chain from data
  to served model. `provenance.baseModelRevision` should equal `baseModelRevisionExpected`.
- **Visibility.** An engineer sees only the pipelines they created; an administrator sees
  all. Enabling a pipeline makes it runnable, not visible in another engineer's list.
- **Lifecycle.** Deleting an enabled pipeline requires disabling it first. A pipeline with run
  history archives rather than hard-deletes.

## Open item — inference serving

DIMER deploys an inference service after training. This pipeline's artifact is an AutoGluon
`TabularPredictor` directory; the `integration` workflow exercises the save → reload → predict
round-trip on a GPU runner, so the artifact is verified to reload and serve predictions outside
the training process. Whether DIMER's own inference-serving layer wraps a tabular predictor — as
opposed to a vision model — is **not yet verified on the platform.** The platform team owns this
check before the pipeline serves production traffic.

## References

- [README.md](README.md) — pipeline reference: model, dataset, fields, outputs, provenance.
- [TABULAR_CLASSIFICATION_DATASET_SPEC.md](TABULAR_CLASSIFICATION_DATASET_SPEC.md) — dataset
  contract and validator checks.
- Platform pipeline setup guide (in the DIMER workbench documentation).
