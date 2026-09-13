# Weight provenance and local snapshot layout

| Field | Value |
|---|---|
| Upstream model | `autogluon/mitra-classifier` (Hugging Face) |
| Files | `model.safetensors` (302,717,904 bytes), `config.json` (81 bytes) |
| Immutable revision | `c425e9fa0910a6be1c494321792e7ba2a1367b1a` |
| SHA-256 `model.safetensors` | `e06a055e91a3baeffc37f9cf634d9e69a27d904b6686131dc3b702f9c0126b19` |
| SHA-256 `config.json` | `2c96c24dd25f64e92753f6f2ba00cc7833b9923459403dcd8504e8700c0995df` |
| License | Apache-2.0 (upstream checkpoint; `LICENSE` ships next to the weights) |
| Package constants | `MODEL_ID`, `MODEL_REVISION` (alias `PINNED_REVISION`), `MODEL_LICENSE`, `MODEL_KEY = "mitra-classifier"`, `WEIGHTS_SHA256`, `CONFIG_SHA256` in `mitra_pipeline/tutorial_api.py` |

## Local snapshot (fleet scheme, NOTEBOOK_SPEC 1.1 ST3/ST4)

The pinned snapshot lives in `weights/mitra-classifier/` (the `MODEL_KEY`). The committed `dimer-base-manifest.json`
there lists both files' paths, byte sizes and SHA-256 and is the parity anchor the standalone tutorials carry inline;
`model.safetensors` is git-ignored (`weights/**/*.safetensors`) while `config.json`, `LICENSE` and the manifest are
committable. `verify_snapshot` re-hashes every manifest entry and asserts the manifest digests equal the package's
`WEIGHTS_SHA256` / `CONFIG_SHA256`; `stage_missing_files(allow_download=True)` fetches only the entries that are
absent, from the Hub at the immutable revision; `MitraClassificationPipeline.from_pretrained(weights_dir=...)` runs both
and then calls the existing `stage_verified_hf_snapshot`, which copies the verified bytes into an offline Hugging Face
cache layout (`HF_HUB_OFFLINE=1`) so AutoGluon's Mitra resolves exactly the pinned revision without a network path. To
use an offline copy (for example the DIMER wizard's model ZIP), place both files in that directory before running.

```bash
python -c "from mitra_pipeline import stage_missing_files, verify_snapshot; print(stage_missing_files(allow_download=True)); print(verify_snapshot()['files'])"
```

The digest checks establish byte integrity against this repository's pinned identity, not producer authenticity or
model quality. The DIMER fine-tuner image bakes the same files at build time (see `DEPLOYMENT.md`).
