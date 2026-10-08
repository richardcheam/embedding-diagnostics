"""Render bounded C3 tables from recorded endpoints; no vector computations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from embedding_diagnostics.bdd100k import ATTRIBUTE_VOCAB
from embedding_diagnostics.phase_c_numerics import CONDITIONS, resume_results

ATTRIBUTES = ('weather', 'scene', 'timeofday')


def render(root):
    state = json.loads((root/'results.json').read_text())
    freeze_path = root/'freeze.json'
    if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != state['identity']['freeze_sha256']:
        raise ValueError('report freeze identity mismatch')
    freeze = json.loads(freeze_path.read_text())
    state = resume_results(root/'results.json', state['identity'])
    verification = json.loads((root/'verification.json').read_text())
    records = [envelope['payload'] for envelope in state['records']]
    if (freeze['conditions'] != list(CONDITIONS) or
        [r['condition'] for r in records] != freeze['conditions'] or
        verification['identity'] != state['identity'] or
        verification['condition_records'] != len(CONDITIONS)):
        raise ValueError('report requires complete frozen conditions and matching verification')
    count = freeze['validation_count']
    if (verification['validation_rows'] != count or any(
        len(r['query_ids']) != count or len(set(r['query_ids'])) != count or
        r['query_ids'] != records[0]['query_ids'] or r['dimension'] != freeze['dimension']
        for r in records)):
        raise ValueError('report row/dimension alignment mismatch')
    lines = ['# C3: numerical error, neighbour identities and attribute utility', '',
             '[ours] Fixed cache-only exact-search experiment. Native 768d is primary;',
             'the sole supporting stress is unchanged training-fitted mean injection .99.',
             'Mean conditions are compared with their own reference. No inference, image',
             'decoding, ANN or C4. All tables are generated from results.json.', '',
             f"Freeze SHA256: `{state['identity']['freeze_sha256']}`.", '',
             '## Numerical error and identity stability', '', f'[ours] {count} queries; k=10.', '',
             '| Condition | Mean max score error | Max error | Overlap@10 | Changed sets | '
             'Boundary ties | Prospective covered / exceptions | Retrospective covered | '
             'Gallery bytes | Score dtype |',
             '|---|---:|---:|---:|---:|---:|---|---:|---:|---|']
    for r in records:
        s = r['summary']
        prospective = 'n/a' if s['prospective_sufficient_queries'] is None else (
            f"{s['prospective_sufficient_queries']} / {s['prospective_empirical_violations']}")
        lines.append(f"| {r['condition']} | {s['mean_max_score_error']:.3e} | "
                     f"{s['max_score_error']:.3e} | {s['mean_overlap']:.6f} | "
                     f"{s['identity_changed_queries']} | {r['boundary_tie_count']} | "
                     f"{prospective} | {s['retrospective_sufficient_queries']} | "
                     f"{r['gallery_storage_bytes']} | {r['score_dtype']} |")
    lines += ['', '## Attribute utility', '',
              '[ours] Micro P@10; delta in percentage points, relative to same-input reference.',
              '',
              '| Condition | Weather P@10 | Delta pp | Scene P@10 | Delta pp | '
              'Time P@10 | Delta pp |', '|---|---:|---:|---:|---:|---:|---:|']
    for r in records:
        cells = [r['condition']]
        for a in ATTRIBUTES:
            cells += [f"{r['attributes'][a]['p10']:.6f}",
                      f"{100*r['attributes'][a]['delta']:+.4f}"]
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '## Turnover counts', '',
              '[ours] Entering/departing label counts, with no arbitrary neighbour pairing.', '',
              '| Condition | Attribute | Gain / loss / unchanged | Changed sets with unchanged '
              'P@10 | Changed sets with unchanged full label counts |',
              '|---|---|---|---:|---:|']
    for r in records:
        for a in ATTRIBUTES:
            t = r['attributes'][a]
            lines.append(f"| {r['condition']} | {a} | {t['gained_queries']} / "
                         f"{t['lost_queries']} / {t['unchanged_queries']} | "
                         f"{t['identity_changed_p10_unchanged']} | "
                         f"{t['identity_changed_label_counts_unchanged']} |")
    reference = records[0]
    lines += ['', '## Support and floors', '',
              '[ours] Full natural-distribution gallery. Macro retrieval includes classes with',
              'validation support>=10; all gallery rows remain. Full per-class deltas and macro',
              'scores are retained in results.json.', '',
              '| Attribute | Class | Validation support | Reference P@10 |',
              '|---|---|---:|---:|']
    for a in ATTRIBUTES:
        for c, s in reference['attributes'][a]['per_class'].items():
            lines.append(f"| {a} | {ATTRIBUTE_VOCAB[a][int(c)]} | "
                         f"{s['support']} | {s['p10']:.6f} |")
    lines += ['', '| Attribute | sum(p²) chance | Exact self-excluded chance |', '|---|---:|---:|']
    for a in ATTRIBUTES:
        t = reference['attributes'][a]
        lines.append(f"| {a} | {t['chance_sum_p_squared']:.6f} | "
                     f"{t['chance_self_excluded']:.6f} |")
    lines += ['', '## Margin analysis', '',
              '[ours] Tied reference margins stay together; bins are separately defined for',
              'native and mean99. Coverage is an empirical sufficient-condition check,',
              'not rigorous floating-point certification.', '',
              '| Condition | Bin | N | Margin min / max | Overlap | Changed | '
              'Prospective / retrospective covered | Weather / scene / time delta pp |',
              '|---|---:|---:|---|---:|---:|---|---|']
    for r in records:
        for b in r['bin_analysis']:
            deltas = ' / '.join(f"{100*b['p10_deltas'][a]:+.4f}" for a in ATTRIBUTES)
            lines.append(f"| {r['condition']} | {b['bin']} | {b['count']} | "
                         f"{b['minimum']:.3e} / {b['maximum']:.3e} | "
                         f"{b['mean_overlap']:.6f} | {b['changed_queries']} | "
                         f"{b['prospective_sufficient']} / {b['retrospective_sufficient']} | "
                         f"{deltas} |")
    lines += ['', '## Runtime and exceptions', '',
              f"[ours] Completing invocation {verification['wall_seconds']:.3f}s; "
              f"summed condition time {sum(r['wall_seconds'] for r in records):.3f}s; "
              'peak completing-process RSS '
              f"{verification['peak_rss_mib']:.2f} MiB; "
              f"{verification['historical_files_preserved']} historical files preserved.", '',
              '| Condition | Wall seconds | Clipped coordinates / rows | Outside range | '
              'Max arithmetic residual |', '|---|---:|---|---:|---:|']
    for r in records:
        c = r['clipping']
        lines.append(f"| {r['condition']} | {r['wall_seconds']:.3f} | "
                     f"{c['clipped_coordinates']} / {c['clipped_rows']} | "
                     f"{c['outside_calibration_coordinates']} | "
                     f"{r['summary']['max_arithmetic_residual']:.3e} |")
    lines += ['', '[ours] C3 references minus committed C1 P@10 (ordering continuity):', '',
              '| Reference | Weather | Scene | Time |', '|---|---:|---:|---:|']
    for name, deltas in verification['historical_reference_p10_deltas'].items():
        lines.append('| '+name+' | '+' | '.join(f'{deltas[a]:+.9f}' for a in ATTRIBUTES)+' |')
    lines += ['', '[interpretation] Identity stability, unchanged attribute counts and unmeasured',
              'instance semantics are distinct. Numerical bounds are sufficient and can be loose.',
              'Storage bytes exclude reconstructed analysis arrays and do not imply operational',
              'search speedups. No query-seed intervals or post-hoc success threshold is supplied.',
              'See interpretation.md for assessment and verification.json for checks.', '']
    (root/'results.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('experiments/phaseC_c3'))
    render(parser.parse_args().output)
