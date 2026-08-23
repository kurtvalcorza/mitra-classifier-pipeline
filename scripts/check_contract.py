#!/usr/bin/env python3
import hashlib,json,sys
from pathlib import Path
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 root=Path(__file__).resolve().parents[1];mods=[root/"validator"/"contract_hardening.py",root/"finetuner"/"contract_hardening.py"]
 if not all(p.exists() for p in mods) or sha(mods[0])!=sha(mods[1]):print("contract copies drifted");return 1
 data=json.loads((root/"dimer-runtime-contract.json").read_text());expected="tabular_regression" if "regressor" in root.name else "tabular_classification"
 if data.get("schemaVersion")!=1 or data.get("sharedCodeStrategy")!="KEEP_PARITY_COPIES":return 1
 if json.loads((root/"finetuner"/"dimer-pipeline.json").read_text()).get("taskType")!=expected:return 1
 model=next(x for x in data["runtimeInputs"] if x["name"]=="DIMER_MODEL_DIR")
 if model["requirement"]!="unsupported":return 1
 print("Contract OK",expected,sha(mods[0]));return 0
if __name__=="__main__":sys.exit(main())
