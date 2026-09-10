from pathlib import Path

root = Path('.')
deploy = root / 'DEPLOYMENT.md'
validator = root / 'scripts/validate_colab_tutorial.py'

text = deploy.read_text(encoding='utf-8')
old = '''- **`ci`** (every push/PR) — compiles the deployable sources, runs the unit suite, enforces
  shared-code parity and notebook contracts, and self-tests the repository-defined DIMER notebook
  package producer. It deliberately does **not** install AutoGluon or execute the notebooks end to end.
'''
new = '''- **`ci`** (every push/PR) — compiles the deployable sources, runs the unit suite, and enforces
  shared-code parity plus notebook static/regression contracts. It deliberately does **not** install
  AutoGluon or execute the notebooks end to end.
'''
if old not in text:
    raise RuntimeError('stale CI description not found exactly once')
text = text.replace(old, new, 1)
deploy.write_text(text, encoding='utf-8')

text = validator.read_text(encoding='utf-8')
needle = 'require("Two GitHub Actions workflows guard the repo" not in deployment_text, "stale two-workflow claim remains in DEPLOYMENT.md")'
if needle not in text:
    raise RuntimeError('deployment validator anchor missing')
text = text.replace(
    needle,
    needle + '\n    require("self-tests the repository-defined DIMER notebook" not in deployment_text, "stale package-producer CI claim remains in DEPLOYMENT.md")',
    1,
)
validator.write_text(text, encoding='utf-8')
print('final round-2 docs cleanup applied')
