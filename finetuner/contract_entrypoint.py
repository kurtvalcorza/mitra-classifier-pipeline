#!/usr/bin/env python3
from __future__ import annotations
import sys,contract_hardening,train
contract_hardening.install_finetuner(train,"tabular_classification")
if __name__=="__main__":sys.exit(train.main())
