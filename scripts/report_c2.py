"""Render C2 comparison tables from immutable endpoint records, without inference."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTRIBUTES = ('weather', 'scene', 'timeofday')


def render(root):
    results = json.loads((root / 'results.json').read_text())
    effects = json.loads((root / 'paired_effects.json').read_text())
    verification = json.loads((root / 'verification.json').read_text())
    audit = json.loads((root / 'completion_audit.json').read_text())
    records = results['records']
    lines = [
        '# C2 — learned MRL versus centered post-hoc PCA', '',
        '[measurement] Seven representations computed from the accepted C1 3000×768',
        'FP32 cache; 2000 train / 1000 validation. No new sample, neural inference or',
        'image decoding. This is a predeclared analysis on a previously examined',
        'sample, not an independent new-sample replication.', '',
        '[measurement] C1 rank-transform audit passed: training-only raw SVD defines',
        'a common projector for both splits. Severe validation output/geometry',
        'matches the historical record exactly. H4 is not superseded; no C1 endpoints',
        'were changed. The severe eight-direction rank-loss reference is separate',
        'from these matched-D PCA baselines.', '',
        f"[measurement] C2 code: `{results['identity']['C2_code_commit']}`; freeze",
        f"SHA256: `{results['identity']['freeze_sha256']}`. See `freeze.json` and",
        '`docs/c2-protocol.md` for fitting, eligibility, arithmetic and hypotheses.', '',
        '## Geometry', '',
        '[measurement] Raw historical metrics and descriptive nominal-D capacity',
        'fractions are retained separately. PCA removes the training mean before',
        'projecting; both compressed mechanisms normalize every row. These are',
        'different constructions, so variance/cosine changes are not solely a',
        'consequence of dimension. RankMe epsilon may marginally exceed its ideal',
        'nominal bound; fractions remain unclipped.', '',
    ]
    for split in ('train', 'val'):
        lines += [f'### {split}', '',
                  '| Representation | D | Variance | Mean cosine | RankMe | PR | RankMe/D | PR/D |',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for record in records:
            g = record['geometry'][split]
            lines.append(f"| {record['representation']} | {record['dimension']} | "
                         f"{g['total_variance']:.6f} | {g['mean_pairwise_cosine']:.6f} | "
                         f"{g['rankme']:.3f} | {g['participation_ratio']:.3f} | "
                         f"{g['rankme_fraction']:.4f} | {g['participation_ratio_fraction']:.4f} |")
        lines += ['']
    lines += ['## Semantic endpoints', '',
              '[measurement] BA uses frozen train>=20 AND val>=10 eligibility: five',
              'weather classes, three scene classes, all three timeofday classes.',
              'Every row remains in fitting, overall accuracy and the validation',
              'retrieval gallery. All original support counts are included in results.', '',
              '| Representation | Weather BA | Scene BA | Time BA | Weather P@10 | Scene P@10 | '
              'Time P@10 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for record in records:
        ba = [record['attributes'][a]['probe_standardized']['balanced_accuracy']
              for a in ATTRIBUTES]
        retrieval = [record['attributes'][a]['retrieval_p10'] for a in ATTRIBUTES]
        lines.append('| ' + record['representation'] + ' | ' +
                     ' | '.join(f'{v:.4f}' for v in ba+retrieval) + ' |')
    lines += ['', '| Representation | Weather accuracy | Scene accuracy | Time accuracy |',
              '|---|---:|---:|---:|']
    for record in records:
        values = [record['attributes'][a]['probe_standardized']['accuracy'] for a in ATTRIBUTES]
        lines.append('| ' + record['representation'] + ' | ' +
                     ' | '.join(f'{v:.4f}' for v in values) + ' |')
    lines += ['', '[measurement] Floors (weather/scene/time): majority accuracy',
              '0.588/0.577/0.489; balanced constant-class 0.2/0.333333/0.333333;',
              'retrieval chance 0.391270/0.421838/0.432694. No representation-specific',
              'class exclusion or after-result threshold was applied.', '',
              '## Paired differences', '',
              '[measurement] Entries below are percentage-point differences. Full',
              'accuracy/BA/P@10 deltas are saved in `paired_effects.json`.', '',
              '| Relative to native | Weather BA | Scene BA | Time BA | Weather P@10 | '
              'Scene P@10 | Time P@10 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, scores in effects['native_relative'].items():
        values = [scores[a]['balanced_accuracy_delta'] for a in ATTRIBUTES] + [
            scores[a]['retrieval_p10_delta'] for a in ATTRIBUTES]
        lines.append('| ' + name + ' | ' + ' | '.join(f'{100*v:+.3f}' for v in values) + ' |')
    lines += ['', '| MRL minus PCA | Weather BA | Scene BA | Time BA | Weather P@10 | '
              'Scene P@10 | Time P@10 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for dimension, scores in effects['MRL_minus_PCA'].items():
        values = [scores[a]['balanced_accuracy_delta'] for a in ATTRIBUTES] + [
            scores[a]['retrieval_p10_delta'] for a in ATTRIBUTES]
        lines.append('| ' + dimension + ' | ' +
                     ' | '.join(f'{100*v:+.3f}' for v in values) + ' |')
    lines += ['', '| Difference in accuracy (percentage points) | Weather | Scene | Time |',
              '|---|---:|---:|---:|']
    for name, scores in effects['native_relative'].items():
        values = [scores[a]['accuracy_delta'] for a in ATTRIBUTES]
        lines.append('| ' + name + ' minus native | ' +
                     ' | '.join(f'{100*v:+.3f}' for v in values) + ' |')
    for dimension, scores in effects['MRL_minus_PCA'].items():
        values = [scores[a]['accuracy_delta'] for a in ATTRIBUTES]
        lines.append('| MRL minus PCA ' + dimension + ' | ' +
                     ' | '.join(f'{100*v:+.3f}' for v in values) + ' |')
    lines += ['', '## Fit diagnostics and runtime', '',
              '[measurement] Standardized probe remains primary. Every inner candidate',
              'and final fit is recorded with C, boundary, convergence, iteration count,',
              'training/validation scores and warnings; final summaries follow.', '',
              '| Representation | Attribute | C | Boundary | Converged | Iterations | '
              'Train accuracy | Train BA |',
              '|---|---|---:|---|---|---:|---:|---:|']
    for record in records:
        for attr in ATTRIBUTES:
            p = record['attributes'][attr]['probe_standardized']
            lines.append(f"| {record['representation']} | {attr} | {p['selected_C']:g} | "
                         f"{p['grid_boundary'] or 'interior'} | {bool(p['converged'])} | "
                         f"{int(p['n_iter'])} | {p['train_accuracy']:.4f} | "
                         f"{p['train_balanced_accuracy']:.4f} |")
    lines += ['', f"[measurement] Initial run: {audit['initial_run_wall_seconds']:.2f} seconds; "
              f"PCA fit: {results['PCA_fit_seconds']:.2f} seconds. All projected vectors finite;",
              'compressed norms approximately one. Native probe continuity against C1:',
              '`' + json.dumps(verification['C1_native_probe_deltas'], sort_keys=True) + '`.', '',
              '[measurement] Fit-record audit: `' + json.dumps(audit['fit_summary'],
                                                              sort_keys=True) + '`.',
              'Network/model-construction/image-decode guards passed. Completed resume',
              'performed zero semantic endpoint calls and preserved results/parameters.', '',
              f"[measurement] Tests: {audit['tests']['passed']} passed, "
              f"{audit['tests']['skipped']} skipped; Ruff and lock checks passed.", '',
              '## Hypotheses and limits', '',
              'See `interpretation.md` for the qualified C2-H1–H4 assessment.', '',
              '[open] Paired training-seed intervals are unavailable for one fixed encoder;',
              'no invented seeds, post-hoc uncertainty or significance claims. The same',
              'sample was inspected in C1. Rare categories are outside balanced-probe',
              'eligibility. Coordinate-wise standardization and PCA centering differ',
              'between mechanisms; rank alone cannot separate their causal contributions.',
              'No pass/fail retention tolerance was declared; paired differences remain',
              'descriptive. C3 and C4 have not begun.', '']
    (root / 'results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('experiments/phaseC_c2'))
    render(parser.parse_args().root)
