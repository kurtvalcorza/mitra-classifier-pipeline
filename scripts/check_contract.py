#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,sys
from pathlib import Path
EXPECTED_HARDENING_SHA256="d694532cc23366c119e975cf67ba51f3d85f722ae1c0d1711b4071a1cca09393"
EXPECTED_CONTRACT_SHA256="d2dc62fe4f0437941a29140c35e083f39e0eba9db87561d3bda6a36e208d8bb9"
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 root=Path(__file__).resolve().parents[1];mods=[root/"validator"/"contract_hardening.py",root/"finetuner"/"contract_hardening.py"];contract=root/"dimer-runtime-contract.json"
 if any(sha(p)!=EXPECTED_HARDENING_SHA256 for p in mods) or sha(contract)!=EXPECTED_CONTRACT_SHA256:return 1
 data=json.loads(contract.read_text());expected="tabular_regression" if "regressor" in root.name else "tabular_classification"
 if data.get("schemaVersion")!=1 or data.get("sharedCodeStrategy")!="KEEP_PARITY_COPIES" or json.loads((root/"finetuner"/"dimer-pipeline.json").read_text()).get("taskType")!=expected:return 1
 if next(x for x in data["runtimeInputs"] if x["name"]=="DIMER_MODEL_DIR")["requirement"]!="unsupported":return 1
 print("Contract OK",expected,EXPECTED_HARDENING_SHA256);return 0
if __name__=="__main__":sys.exit(main())
