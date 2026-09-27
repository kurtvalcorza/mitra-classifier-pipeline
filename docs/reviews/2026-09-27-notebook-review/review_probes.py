"""Offline probes of exact notebook logic transcribed from pinned source.
Not an execution of the full notebook or of its foundation-model dependencies.
"""
import json, tempfile, zipfile
from pathlib import Path, PurePosixPath
import numpy as np
import sklearn
from sklearn.metrics import log_loss

# Verbatim from notebook cell xUUykd5OFXEc, excluding type annotation only.
def align_probabilities(raw, model_classes, classes) -> np.ndarray:
    array = np.asarray(raw, dtype=float)
    source = [str(value) for value in model_classes]
    target = [str(value) for value in classes]
    if array.ndim != 2 or array.shape[1] != len(source):
        raise ValueError("Probability array shape does not match the model class list.")
    mapping = {label: i for i, label in enumerate(source)}
    missing = [label for label in target if label not in mapping]
    if missing:
        raise ValueError(f"Model probabilities omit required classes: {missing}")
    aligned = array[:, [mapping[label] for label in target]]
    if not np.isfinite(aligned).all():
        raise ValueError("Probabilities contain non-finite values.")
    row_sums = aligned.sum(axis=1)
    if not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError("Probability rows do not sum to approximately 1.")
    # Remove float32 rounding so every row sums to exactly 1 before scoring.
    return aligned / row_sums[:, None]

# Exact staging block from cell ZDCzBnsvFXEe, wrapped as a callable for tests.
def stage_archive(archive_path, DATA_ROOT):
    required_members = {"train.csv", "val.csv", "test.csv"}
    staged_paths = {}
    extract_root = DATA_ROOT / "staged"
    extract_root.mkdir(exist_ok=False)
    with zipfile.ZipFile(archive_path) as zf:
        selected = {}
        expanded_bytes = 0
        for info in zf.infolist():
            name = info.filename
            posix = PurePosixPath(name)
            if "\\" in name or posix.is_absolute() or ".." in posix.parts:
                raise ValueError(f"Unsafe ZIP path: {name}")
            expanded_bytes += info.file_size
            if expanded_bytes > 512 * 1024 * 1024:
                raise ValueError("Dataset archive exceeds the 512 MiB expanded-size ceiling.")
            if not info.is_dir() and posix.name in required_members:
                if posix.name in selected:
                    raise ValueError(f"Duplicate split member: {posix.name}")
                selected[posix.name] = info
        missing = required_members - set(selected)
        if missing:
            raise ValueError(f"Dataset is missing required split files: {sorted(missing)}")
        for member_name, info in selected.items():
            target = extract_root / member_name
            target.write_bytes(zf.read(info))
            staged_paths[member_name[:-4]] = target
    return staged_paths

results=[]
def record(name, fn):
    try:
        result=fn()
        results.append({'probe':name,'result':str(result),'raised':None})
    except Exception as exc:
        results.append({'probe':name,'result':None,'raised':type(exc).__name__,'message':str(exc)})

labels=['low','mid','high']
record('correct semantic probability reordering',lambda:align_probabilities([[.7,.2,.1]],['high','low','mid'],labels).tolist())
record('reject missing class',lambda:align_probabilities([[.7,.3]],['low','mid'],labels).tolist())
record('reject non-finite probabilities',lambda:align_probabilities([[np.nan,.3,.7]],labels,labels).tolist())
record('accepts negative probability with unit row sum',lambda:align_probabilities([[-.2,.6,.6]],labels,labels).tolist())
record('log_loss invalid probability behavior in installed sklearn',lambda:log_loss([0,1,2],np.array([[-.2,.6,.6],[.1,.8,.1],[.1,.1,.8]]),labels=[0,1,2]))
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp)
    valid=root/'valid.zip';invalid=root/'missing-test.zip';unsafe=root/'unsafe.zip';duplicate=root/'duplicate.zip'
    for path,names in [(valid,['train.csv','val.csv','test.csv']),(invalid,['train.csv','val.csv']),(unsafe,['../train.csv','val.csv','test.csv']),(duplicate,['train.csv','another/train.csv','val.csv','test.csv'])]:
        with zipfile.ZipFile(path,'w') as z:
            for name in names:z.writestr(name,'a,target\n1,low\n')
    for name in ['normal','retry','unsafe','duplicate']:(root/name).mkdir()
    record('first valid archive staging',lambda:list(stage_archive(valid,root/'normal')))
    record('repeat same acquisition cell',lambda:list(stage_archive(valid,root/'normal')))
    record('invalid archive missing test',lambda:stage_archive(invalid,root/'retry'))
    record('retry corrected archive in same session',lambda:list(stage_archive(valid,root/'retry')))
    record('reject path traversal',lambda:stage_archive(unsafe,root/'unsafe'))
    record('reject duplicate split',lambda:stage_archive(duplicate,root/'duplicate'))
out={'scope':'Transcribed notebook logic; synthetic inputs; no model downloads or Colab run','sklearn':sklearn.__version__,'numpy':np.__version__,'probes':results}
Path(__file__).with_name('probe_results.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out,indent=2))
