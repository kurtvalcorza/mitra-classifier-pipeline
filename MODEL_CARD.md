---
license: apache-2.0
pipeline_tag: tabular-classification
tags:
  - tabular-classification
  - tabular-foundation-model
  - in-context-learning
base_model: autogluon/mitra-classifier
---

# Mitra classifier — model card (for the DIMER pipeline)

**This is not a model this project trained.** Mitra was pretrained by the AutoGluon team at
AWS and is used here unmodified under Apache-2.0. This card records what the pipeline uses,
where it comes from, and how it behaves; see the [README](README.md) for the pipeline itself.

## Summary

| | |
|---|---|
| Base model | [`autogluon/mitra-classifier`](https://huggingface.co/autogluon/mitra-classifier) |
| Pinned revision | `c425e9fa0910a6be1c494321792e7ba2a1367b1a` |
| Architecture | 12-layer Transformer, 512 embedding, 4 heads, row + column (2D) attention |
| Parameters | ~75.7M (Hugging Face metadata, F32) |
| Pretraining | 45M synthetic datasets on 8×A100 (~60 h); no real data seen |
| Task | tabular classification (categorical target; binary or multiclass, up to 10 classes) |
| Licence | Apache-2.0 — redistribution and hosted serving permitted, including commercial |

`model.safetensors` is 302,717,904 bytes, SHA-256
`e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19`; `config.json` is 86 bytes,
SHA-256 `2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df`.

## Providing the weights to DIMER

The weights are **not committed to this repository** (289 MB; and DIMER's build may not fetch
Git LFS). A local copy is kept in `weights/` (gitignored — never pushed) for convenience: pick
`weights/model.safetensors` in the wizard's upload box, or use it for a local image bake.
Provide them to the pipeline in one of three ways:

1. **HuggingFace Model ID (default).** Enter `autogluon/mitra-classifier` as the Base Model;
   AutoGluon downloads it at runtime. Requires egress to `huggingface.co`.
2. **Upload weights (no egress).** Use the wizard's *upload your own model weights* box with
   `model.safetensors`; DIMER stores it in S3 and the fine-tuner reads it at runtime.
3. **Bake into the image.** Uncomment Option A in the fine-tuner `Dockerfile` to download the
   pinned revision at build time (needs egress at build), or `COPY` a local copy in.

Fetch the pinned bytes with the Hugging Face CLI:

```bash
hf download autogluon/mitra-classifier model.safetensors config.json \
  --revision c425e9fa0910a6be1c494321792e7ba2a1367b1a --local-dir .
```

Every fine-tuning run records the revision it actually used in `result.json`
(`provenance.baseModelRevision`) and compares it to the pinned value.

## How the pipeline uses it

- **Fine-tune (GPU)** — adapts the pretrained weights to the uploaded table. Requires a GPU.
- **Zero-shot (CPU)** — in-context inference with no weight update; used automatically when no
  GPU is available (Mitra's fine-tuning backward pass is unsupported on many CPUs).

The fine-tuner infers the problem type from the target's cardinality — `binary` for two
classes, `multiclass` for three to ten — so AutoGluon loads the classifier checkpoint. It
selects fine-tune versus zero-shot from GPU availability at runtime and records both the mode
and device in `result.json` (`metrics.mode`, `metrics.device`, `metrics.problemType`,
`metrics.numClasses`). See the [README](README.md).

## Applicability and limits

Mitra is strongest on **small tabular data** (below ~5,000 samples and ~100 features). Hard
limits: **10,000 training rows, 500 features, 10 classes**. It needs about **8.7 GB of memory**
(measured on 6,400 rows); request a profile that clears that with headroom (see the README's
resource profile).

## Measured behaviour in this pipeline

Small smoke-test runs on the bundled sample dataset (a 3-class demand band derived from
FreshRetailNet-50K; 6,400 train / 1,600 validation rows, one seed, one split). These size and
exercise the pipeline; they are **not a benchmark**. The majority-class baseline predicts the
most frequent training class for every row.

| Dataset | Classes | Mode | Mitra accuracy | Majority-class baseline |
|---|---|---|---|---|
| FreshRetailNet demand band | 3 | fine-tune (GPU) | **0.576** | 0.361 |
| FreshRetailNet demand band | 3 | zero-shot (CPU) | 0.589 | 0.361 |

Both modes beat the majority-class baseline by more than 0.2 accuracy, so the features carry
signal about the demand band. Zero-shot matched fine-tuning on this small 3-class table — on
larger or harder tables fine-tuning is expected to help more. Read these as direction, not
scores. Treat Mitra's published benchmark results as evidence of strong performance where
signal exists, not a guarantee on any table.

## Licence

Apache-2.0. Redistribution and hosted serving are both permitted, including commercial use, on
the conditions of retaining the `LICENSE` and stating modifications (there are none). The
Apache-2.0 licence text ships with the weights upstream.

## Citation

Cite the original work, not this repository:

> Zhang, X., Maddix, D. C., Yin, J., Erickson, N., Ansari, A. F., Han, B., Zhang, S., Akoglu,
> L., Faloutsos, C., Mahoney, M., Hu, T., Rangwala, H., Karypis, G., & Wang, Y. (2025).
> *Mitra: Mixed Synthetic Priors for Enhancing Tabular Foundation Models.* NeurIPS 2025.
> arXiv:2510.21204. https://doi.org/10.48550/arXiv.2510.21204
