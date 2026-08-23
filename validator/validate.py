#!/usr/bin/env python3
from __future__ import annotations
import sys,contract_hardening,validator
contract_hardening.install_validator(validator,"tabular_classification")
if __name__=="__main__":sys.exit(validator.main())
