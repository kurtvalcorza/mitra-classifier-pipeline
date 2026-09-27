"""Isolated control-flow reproductions from the reviewed notebook.

These are deliberately small synthetic probes, not executions of the notebook,
its adapters, the real dataset, or a Colab/GPU model run.
"""
from __future__ import annotations
import hashlib
import json
import tempfile
from pathlib import Path
import numpy as np

REVISION = '6c9f911294712eccf993817ef7c48c9f342bc194'
results = []

# Section 7.1 constructs condition_override by appending the ablation group.
# Section 5.3, run_tabicl_development, dispatches on exact string equality.
for base_condition in ('in_context', 'fine_tuned'):
    for group in ('no_stockout', 'no_period_proxies'):
        condition_override = base_condition + '_' + group
        base_uses_finetuning = base_condition == 'fine_tuned'
        ablation_uses_finetuning = condition_override == 'fine_tuned'
        results.append({
            'probe': 'TabICLv2 ablation adaptation dispatch',
            'base_condition': base_condition,
            'ablation_condition': condition_override,
            'base_uses_finetuning': base_uses_finetuning,
            'ablation_uses_finetuning': ablation_uses_finetuning,
            'adaptation_mode_preserved': base_uses_finetuning == ablation_uses_finetuning,
        })

# Section 8.1 verifies the development run_config.json file hash but constructs
# the test's model_configuration from the current CONDITION_SPECS mapping.
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / 'run_config.json'
    path.write_text(json.dumps({'model_configuration': {'n_ensembles': 4, 'context_size': 4180, 'batch_size': 4096}}))
    original_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    live_condition_specs = {'tabdpt_icl': {'configuration': {'n_ensembles': 8, 'context_size': 4180, 'batch_size': 4096}}}
    verified = hashlib.sha256(path.read_bytes()).hexdigest() == original_hash
    test_configuration = live_condition_specs['tabdpt_icl']['configuration']
    frozen_configuration = json.loads(path.read_text())['model_configuration']
    results.append({
        'probe': 'Unchanged development config file does not bind live test configuration',
        'persisted_development_config_hash_check_passes': verified,
        'persisted_development_configuration': frozen_configuration,
        'test_configuration_taken_from_live_mapping': test_configuration,
        'configuration_matches_development': test_configuration == frozen_configuration,
    })

# Synthetic counterexample: an interval below zero versus RF does not imply
# an interval below zero versus full-feature LightGBM. Here full and ablated
# predictions are identical; both are worse than the separate reference.
def balanced_accuracy(y, pred):
    return float(np.mean([np.mean(pred[y == c] == c) for c in np.unique(y)]))

truth = np.tile(np.arange(3), 100)
reference_pred = truth.copy()
full_pred = np.zeros_like(truth)
ablated_pred = full_pred.copy()
rng = np.random.default_rng(42)
draws = rng.integers(0, len(truth), size=(200, len(truth)))
gain_vs_reference = []
gain_vs_full = []
for rows in draws:
    a = balanced_accuracy(truth[rows], ablated_pred[rows])
    gain_vs_reference.append(a - balanced_accuracy(truth[rows], reference_pred[rows]))
    gain_vs_full.append(a - balanced_accuracy(truth[rows], full_pred[rows]))
results.append({
    'probe': 'Synthetic illustration: paired intervals are specific to their comparator',
    'ablated_minus_separate_reference_95pct_interval': np.percentile(gain_vs_reference, [2.5, 97.5]).tolist(),
    'ablated_minus_full_same_model_95pct_interval': np.percentile(gain_vs_full, [2.5, 97.5]).tolist(),
    'full_and_ablated_predictions_identical': bool(np.array_equal(full_pred, ablated_pred)),
})

payload = {'revision': REVISION, 'scope': __doc__.strip(), 'results': results}
out = Path(__file__).with_name('control_flow_results.json')
out.write_text(json.dumps(payload, indent=2) + '\n')
print(json.dumps(payload, indent=2))
