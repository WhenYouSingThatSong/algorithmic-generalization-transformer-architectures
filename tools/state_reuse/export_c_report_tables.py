#!/usr/bin/env python3
"""Re-export C functional-probe report tables from completed raw records.

CPU only: reads the fixed arm record files, recomputes the q last-two
VALUE/first-four outcomes and node-level reference-correct/damage counts,
and writes the small tables used by docs/records/2026-11-19/c-state-reuse.
It does not load weights, run the model, or modify the source run directory.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'priority_pilots'))
import core as P
import summarize as S
from io_utils import jsonl, read_json, sha, write_json

ARMS = ('initial', 'hold', 'noise', 'semantic')
TIMINGS = ('early', 'hold', 'late')
MODES = ('natural', 'same', 'semantic', 'noise2')
LAGS = (1, 2, 4)
BRANCHES = P.C_BRANCHES
CONTRASTS = (('clean_keep', 'clean_pre'), ('observed_keep', 'observed_pre'),
             ('observed_keep', 'restore_u_pre'), ('observed_keep', 'clean_pre'))
REGIONS = ('q', 'other_children', 'descendants_excl_q', 'outside')
CTX_FIELDS = ('clean_local_first4_correct', 'u_value_correct', 'other_parents_value_correct',
              'q_value_correct', 'donor_pre_empty', 'donor_pre_identity_correct',
              'donor_solved_first4_correct')


def validate(bundle):
    meta = read_json(bundle / 'C-summary/meta.json')
    if meta.get('status') != 'complete' or meta.get('protocol') != 'state_reuse_summary_v1':
        raise ValueError('complete C summary required')
    for arm in ARMS:
        run = bundle / f'C-{arm}'
        m = read_json(run / 'meta.json')
        if m.get('status') != 'complete' or m.get('protocol') != 'C_function_probe_v1':
            raise ValueError(f'incomplete arm {arm}')
        if sha(run / 'records.jsonl') != m['records_sha256']:
            raise ValueError(f'records changed: {arm}')
        if sha(run / 'coverage.jsonl') != m['coverage_sha256']:
            raise ValueError(f'coverage changed: {arm}')
    return meta


def record_meta(meta):
    return {s['meta']['models'][0]['arm']: s for s in meta['sources']}


def q_correct(r, branch):
    qi = r['positions'].index(r['q'])
    gold = r['truth4'][qi]
    outs = r['branches'][branch]['outputs'][-2:]
    return (all(z[qi][3] == gold[3] for z in outs),
            all(z[qi] == gold for z in outs))


def region_positions(r):
    q, u = r['q'], r['u']
    return dict(q=[q],
                other_children=[p for p in r['children'] if p != q],
                descendants_excl_q=[p for p in r['descendants'] if p != q],
                outside=[p for p in r['positions'] if p not in set(r['descendants'] + [u])])


def branch_ok(outputs, gold, i, metric):
    if metric == 'value':
        return all(z[i][3] == gold[i][3] for z in outputs)
    return all(z[i] == gold[i] for z in outputs)


def qualified_sets(bundle):
    qsets = {arm: set() for arm in ARMS}
    for arm in ARMS:
        for r in jsonl(bundle / f'C-{arm}' / 'records.jsonl'):
            if S.input_qualified(r):
                qsets[arm].add((r['graph_id'], r['timing'], r['mode'], r['probe_lag']))
    return qsets


def collect(bundle, common, candidate_ids):
    qual_counts = defaultdict(int)
    records_counts = defaultdict(int)
    ctx_counts = defaultdict(lambda: defaultdict(int))
    q_out = defaultdict(lambda: [0, 0, 0])
    recovery = defaultdict(lambda: defaultdict(int))
    node_counts = defaultdict(lambda: [0, 0])
    unique_damage = {}
    examples = {}
    arm_counts = defaultdict(lambda: defaultdict(int))
    arm_graphs = defaultdict(lambda: defaultdict(set))
    checks = defaultdict(int)
    cleanpath_graphs = set()
    for arm in ARMS:
        for r in jsonl(bundle / f'C-{arm}' / 'records.jsonl'):
            key = (arm, r['timing'], r['mode'], r['probe_lag'])
            records_counts[key] += 1
            checks['records'] += 1
            arm_counts[arm]['records'] += 1
            if r['branches']['sham'] == r['branches']['observed_keep']:
                checks['sham_equals_observed_keep'] += 1
            for f in CTX_FIELDS:
                ctx_counts[key][f] += int(r['context'][f])
            qvals = {b: q_correct(r, b) for b in BRANCHES}
            checks['clean_pre_value_eq_clean_keep'] += int(
                qvals['clean_pre'][0] == qvals['clean_keep'][0])
            checks['clean_pre_first4_eq_clean_keep'] += int(
                qvals['clean_pre'][1] == qvals['clean_keep'][1])
            ok = S.input_qualified(r)
            if ok:
                qual_counts[key] += 1
            common_here = (r['graph_id'], r['timing'], r['mode'], r['probe_lag']) in common
            cohorts = [('all_recorded', True)]
            if ok:
                cohorts += [('input_qualified', True)]
            if common_here:
                cohorts += [('common_input_qualified', True)]
            for cohort, _ in cohorts:
                for b in BRANCHES:
                    v, f = qvals[b]
                    z = q_out[(arm, r['timing'], r['mode'], r['probe_lag'], cohort, b)]
                    z[0] += 1
                    z[1] += int(v)
                    z[2] += int(f)
            if not ok:
                continue
            ctx_counts[key]['clean_pre_last_two_correct'] += int(qvals['clean_pre'][1])
            ctx_counts[key]['candidate_records'] += 1
            a = arm_counts[arm]
            a['input_qualified'] += 1
            for b in ('observed_keep', 'observed_pre', 'observed_solved', 'restore_u_pre',
                      'clean_keep', 'clean_pre', 'clean_solved'):
                a[b + '_correct'] += int(qvals[b][0])
            for b in ('observed_keep', 'observed_pre', 'clean_keep', 'clean_pre'):
                a['first4_' + b + '_correct'] += int(qvals[b][1])
            kv, kf = qvals['observed_keep']
            ck, ckf = qvals['clean_keep']
            if not kv:
                a['damage'] += 1
                arm_graphs[arm]['damage'].add(r['graph_id'])
                if not ck:
                    a['cleanpath'] += 1
                    arm_graphs[arm]['cleanpath'].add(r['graph_id'])
                    cleanpath_graphs.add((arm, r['graph_id']))
                else:
                    a['perturb_damage'] += 1
                    arm_graphs[arm]['perturb'].add(r['graph_id'])
                    recovery[key]['perturb_damage'] += 1
                    for b, name in (('observed_pre', 'pre_fix'), ('restore_u_pre', 'ru_fix'),
                                    ('clean_pre', 'cp_fix'), ('observed_solved', 'solved_fix')):
                        if qvals[b][0]:
                            a[name] += 1
                            arm_graphs[arm][name].add(r['graph_id'])
                            recovery[key][name] += 1
                recovery[key]['damage'] += 1
                if not ck:
                    recovery[key]['cleanpath'] += 1
                if r['context']['q_value_correct']:
                    a['qinput_ok_damage'] += 1
                    recovery[key]['qinput_ok_damage'] += 1
                else:
                    a['qinput_bad_damage'] += 1
                    recovery[key]['qinput_bad_damage'] += 1
            # node-level reference-correct / branch-wrong counts
            regions = region_positions(r)
            for metric in ('value', 'first4'):
                for ref, br in CONTRASTS:
                    for reg, ps in regions.items():
                        for p in ps:
                            i = r['positions'].index(p)
                            if branch_ok(r['branches'][ref]['outputs'][-2:], r['truth4'], i, metric):
                                z = node_counts[(arm, r['timing'], r['mode'], r['probe_lag'],
                                                 metric, ref, br, reg)]
                                z[0] += 1
                                if not branch_ok(r['branches'][br]['outputs'][-2:], r['truth4'], i, metric):
                                    z[1] += 1
                                    k = (metric, ref, br, arm, r['graph_id'], p, reg)
                                    if k not in unique_damage:
                                        unique_damage[k] = (r['timing'], r['mode'], r['probe_lag'])
            # Fixed-order examples: arms/timing/mode/lag/graph ascending.
            order = (ARMS.index(arm), TIMINGS.index(r['timing']), MODES.index(r['mode']),
                     r['probe_lag'], r['graph_id'])
            keep_v = qvals['observed_keep'][0]; pre_v = qvals['observed_pre'][0]
            ru_v = qvals['restore_u_pre'][0]; cp_v = qvals['clean_pre'][0]
            keep_f = qvals['observed_keep'][1]; pre_f = qvals['observed_pre'][1]
            category = None
            if keep_v == 0 and ck == 1 and not r['context']['q_value_correct'] and ru_v == 0:
                category = 'A_observed_receiver'
            elif keep_v == 0 and pre_v == 0 and ru_v == 1 and ck == 1:
                category = 'B_restore_u_lag1'
            elif keep_v == 0 and ck == 0:
                category = 'C_late_clean_drift'
            elif keep_v == 1 and keep_f == 0 and pre_f == 1 and cp_v == 1:
                category = 'D_format_only'
            if category and (category not in examples or order < examples[category][0]):
                examples[category] = (order, r)
    checks['damage_records'] = sum(a['damage'] for a in arm_counts.values())
    checks['cleanpath_records'] = sum(a['cleanpath'] for a in arm_counts.values())
    checks['cleanpath_graphs'] = len(cleanpath_graphs)
    checks['perturb_damage_records'] = sum(a['perturb_damage'] for a in arm_counts.values())
    checks['common_input_qualified'] = len(common)
    return (qual_counts, records_counts, ctx_counts, q_out, recovery, node_counts,
            unique_damage, examples, arm_counts, arm_graphs, checks)


def equation(r, node):
    pos = r['positions']; lut = {p: i for i, p in enumerate(pos)}
    ps = r['parents'].get(str(node), r['parents'].get(node, []))
    if not ps:
        return f'x{node}={r["truth4"][lut[node]][3]}'
    return 'x%d=%s=%d' % (node, '+'.join('x%d(%d)' % (p, r['truth4'][lut[p]][3]) for p in ps),
                          r['truth4'][lut[node]][3])


def example_row(category, r):
    pos = r['positions']; lut = {p: i for i, p in enumerate(pos)}
    u, q = r['u'], r['q']
    vals = []
    for b in ('observed_keep', 'observed_pre', 'restore_u_pre', 'clean_pre', 'clean_keep'):
        vals += list(q_correct(r, b))
    return dict(category=category, arm=r['arm'], timing=r['timing'], mode=r['mode'],
                lag=r['probe_lag'], graph_id=r['graph_id'], u=u, q=q,
                state_round=r['state_round'], absolute_round=r['absolute_round'],
                u_truth4=json.dumps(r['truth4'][lut[u]]),
                q_truth4=json.dumps(r['truth4'][lut[q]]),
                u_equation=equation(r, u), q_equation=equation(r, q),
                q_value_correct_at_probe=r['context']['q_value_correct'],
                keep_value=vals[0], keep_first4=vals[1], pre_value=vals[2], pre_first4=vals[3],
                rupre_value=vals[4], rupre_first4=vals[5], cleanpre_value=vals[6], cleanpre_first4=vals[7],
                cleankeep_value=vals[8], cleankeep_first4=vals[9])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args()
    summary_meta = validate(a.bundle)
    meta_by_arm = record_meta(summary_meta)
    candidate_ids = meta_by_arm[ARMS[0]]['meta']['candidate_ids']
    qsets = qualified_sets(a.bundle)
    common = set.intersection(*qsets.values())
    (qual_counts, records_counts, ctx_counts, q_out, recovery, node_counts,
     unique_damage, examples, arm_counts, arm_graphs, checks) = collect(
        a.bundle, common, candidate_ids)
    a.out_dir.mkdir(parents=True, exist_ok=True)

    with (a.out_dir / 'c-q-tenway.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'timing', 'mode', 'lag', 'cohort', 'branch', 'n', 'value_n', 'first4_n'])
        for arm in ARMS:
            for t in TIMINGS:
                for m in MODES:
                    for lag in LAGS:
                        for cohort in ('input_qualified', 'common_input_qualified', 'all_recorded'):
                            for b in BRANCHES:
                                z = q_out.get((arm, t, m, lag, cohort, b))
                                if z:
                                    w.writerow([arm, t, m, lag, cohort, b, *z])

    branch_cols = [f'iq_{b}_{x}' for b in BRANCHES for x in ('value', 'first4')]
    ar_cols = [f'ar_{b}_{x}' for b in ('observed_keep', 'observed_pre', 'restore_u_pre',
                                       'clean_pre', 'clean_keep') for x in ('value', 'first4')]
    with (a.out_dir / 'c-conditions.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'timing', 'mode', 'lag', 'candidate_records', 'input_qualified',
                    'common_input_qualified', *CTX_FIELDS, 'clean_pre_last_two_correct',
                    'damage', 'cleanpath', 'perturb_damage', 'qinput_ok_damage',
                    'qinput_bad_damage', 'pre_fix', 'ru_fix', 'cp_fix', 'solved_fix',
                    *branch_cols, *ar_cols])
        for arm in ARMS:
            for t in TIMINGS:
                for m in MODES:
                    for lag in LAGS:
                        key = (arm, t, m, lag)
                        ctx = ctx_counts.get(key, {})
                        rec = recovery.get(key, {})
                        row = [arm, t, m, lag, records_counts[key], qual_counts[key],
                               sum(1 for gid in candidate_ids if (gid, t, m, lag) in common),
                               *[ctx.get(f, 0) for f in CTX_FIELDS],
                               q_out.get((arm, t, m, lag, 'all_recorded', 'clean_pre'), [0, 0, 0])[2],
                               rec.get('damage', 0), rec.get('cleanpath', 0),
                               rec.get('perturb_damage', 0), rec.get('qinput_ok_damage', 0),
                               rec.get('qinput_bad_damage', 0), rec.get('pre_fix', 0),
                               rec.get('ru_fix', 0), rec.get('cp_fix', 0),
                               rec.get('solved_fix', 0)]
                        for b in BRANCHES:
                            z = q_out.get((arm, t, m, lag, 'input_qualified', b), [0, 0, 0])
                            row += [z[1], z[2]]
                        for b in ('observed_keep', 'observed_pre', 'restore_u_pre',
                                  'clean_pre', 'clean_keep'):
                            z = q_out.get((arm, t, m, lag, 'all_recorded', b), [0, 0, 0])
                            row += [z[1], z[2]]
                        w.writerow(row)

    with (a.out_dir / 'c-condition-recovery.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'timing', 'mode', 'lag', 'records', 'qualified', 'damage', 'cleanpath',
                    'perturb_damage', 'qinput_ok_damage', 'qinput_bad_damage', 'pre_fix', 'ru_fix',
                    'cp_fix', 'solved_fix'])
        for arm in ARMS:
            for t in TIMINGS:
                for m in MODES:
                    for lag in LAGS:
                        rec = recovery.get((arm, t, m, lag), {})
                        w.writerow([arm, t, m, lag, records_counts[(arm, t, m, lag)],
                                    qual_counts[(arm, t, m, lag)], rec.get('damage', 0),
                                    rec.get('cleanpath', 0), rec.get('perturb_damage', 0),
                                    rec.get('qinput_ok_damage', 0), rec.get('qinput_bad_damage', 0),
                                    rec.get('pre_fix', 0), rec.get('ru_fix', 0),
                                    rec.get('cp_fix', 0), rec.get('solved_fix', 0)])

    with (a.out_dir / 'c-node-region-counts.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['arm', 'timing', 'mode', 'lag', 'metric', 'reference', 'branch', 'region',
                    'correct_n', 'damage_n'])
        for k, v in sorted(node_counts.items()):
            w.writerow([*k, *v])
    with (a.out_dir / 'c-unique-damage.csv').open('w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['metric', 'reference', 'branch', 'arm', 'graph_id', 'node', 'region',
                    'first_timing', 'first_mode', 'first_lag'])
        for k, cond in sorted(unique_damage.items()):
            w.writerow([*k, *cond])

    if examples:
        first = next(iter(sorted(examples)))
        with (a.out_dir / 'c-examples.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(example_row(first, examples[first][1])))
            w.writeheader()
            for cat in sorted(examples):
                w.writerow(example_row(cat, examples[cat][1]))

    arm_rows = []
    for arm in ARMS:
        arow = arm_counts[arm]
        row = dict(arm=arm, records=arow.get('records', 0),
                   input_qualified=arow.get('input_qualified', 0),
                   common_input_qualified=len(common),
                   damage=arow.get('damage', 0), damage_graphs=len(arm_graphs[arm]['damage']),
                   cleanpath=arow.get('cleanpath', 0),
                   cleanpath_graphs=len(arm_graphs[arm]['cleanpath']),
                   perturb_damage=arow.get('perturb_damage', 0),
                   perturb_graphs=len(arm_graphs[arm]['perturb']),
                   qinput_ok_damage=arow.get('qinput_ok_damage', 0),
                   qinput_bad_damage=arow.get('qinput_bad_damage', 0),
                   pre_fix=arow.get('pre_fix', 0), pre_fix_graphs=len(arm_graphs[arm]['pre_fix']),
                   ru_fix=arow.get('ru_fix', 0), ru_fix_graphs=len(arm_graphs[arm]['ru_fix']),
                   cp_fix=arow.get('cp_fix', 0), cp_fix_graphs=len(arm_graphs[arm]['cp_fix']),
                   solved_fix=arow.get('solved_fix', 0),
                   keep_correct=arow.get('observed_keep_correct', 0),
                   pre_correct=arow.get('observed_pre_correct', 0),
                   solved_correct=arow.get('observed_solved_correct', 0),
                   ru_pre_correct=arow.get('restore_u_pre_correct', 0),
                   clean_keep_correct=arow.get('clean_keep_correct', 0),
                   clean_pre_correct=arow.get('clean_pre_correct', 0),
                   clean_solved_correct=arow.get('clean_solved_correct', 0),
                   first4_keep_correct=arow.get('first4_observed_keep_correct', 0),
                   first4_pre_correct=arow.get('first4_observed_pre_correct', 0),
                   first4_clean_keep_correct=arow.get('first4_clean_keep_correct', 0),
                   first4_clean_pre_correct=arow.get('first4_clean_pre_correct', 0))
        arm_rows.append(row)
    cols = ['arm', 'records', 'input_qualified', 'common_input_qualified', 'damage',
            'damage_graphs', 'cleanpath', 'cleanpath_graphs', 'perturb_damage',
            'perturb_graphs', 'qinput_ok_damage', 'qinput_bad_damage', 'pre_fix',
            'pre_fix_graphs', 'ru_fix', 'ru_fix_graphs', 'cp_fix', 'cp_fix_graphs',
            'solved_fix', 'keep_correct', 'pre_correct', 'solved_correct', 'ru_pre_correct',
            'clean_keep_correct', 'clean_pre_correct', 'clean_solved_correct',
            'first4_keep_correct', 'first4_pre_correct', 'first4_clean_keep_correct',
            'first4_clean_pre_correct']
    with (a.out_dir / 'c-arm-summary.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
        w.writeheader(); w.writerows(arm_rows)

    files = {}
    for name in ('c-q-tenway.csv', 'c-conditions.csv', 'c-condition-recovery.csv',
                 'c-node-region-counts.csv', 'c-unique-damage.csv', 'c-examples.csv',
                 'c-arm-summary.csv'):
        files[name] = sha(a.out_dir / name)
    plan = read_json(a.bundle / 'plan.json')
    complete = read_json(a.bundle / 'complete.json')
    audit = dict(
        date='2026-11-19', protocol='C_function_probe_v1',
        bundle=str(a.bundle), summary_meta_sha256=sha(a.bundle / 'C-summary/meta.json'),
        plan_sha256=complete.get('plan_sha256'),
        base_commit=plan.get('base_commit'),
        complete=complete,
        summary_meta=summary_meta,
        sources={arm: dict(path=str(a.bundle / f'C-{arm}'),
                           meta_sha256=sha(a.bundle / f'C-{arm}' / 'meta.json'),
                           records_sha256=meta_by_arm[arm]['meta']['records_sha256'],
                           coverage_sha256=meta_by_arm[arm]['meta']['coverage_sha256'],
                           models=meta_by_arm[arm]['meta']['models'],
                           code_hashes=meta_by_arm[arm]['meta']['code_hashes'],
                           data_sha256=meta_by_arm[arm]['meta']['data_sha256'],
                           plan_sha256=meta_by_arm[arm]['meta']['plan_sha256'])
                 for arm in ARMS},
        denominators=dict(records=sum(r['records'] for r in arm_rows),
                          input_qualified=sum(r['input_qualified'] for r in arm_rows),
                          common_input_qualified=len(common),
                          per_arm={r['arm']: {k: r.get(k, 0) for k in
                                              ('records', 'input_qualified', 'damage', 'cleanpath',
                                               'perturb_damage', 'qinput_ok_damage', 'qinput_bad_damage',
                                               'pre_fix', 'ru_fix', 'cp_fix', 'solved_fix')}
                                   for r in arm_rows}),
        checks=dict(checks,
                    unique_node_damage=len({(k[3], k[4], k[5]) for k in unique_damage})),
        examples=dict(rule='arms initial<hold<noise<semantic; timing early<hold<late; '
                           'mode natural<same<semantic<noise2; lag 1<2<4; graph_id ascending; '
                           'first qualified record matching each category',
                      categories=sorted(examples)),
        files=files,
        note='VALUE is the primary arithmetic field; first4 is the stricter four-field readout. '
             'The automatic C delivery uses all_recorded and first4; its late/semantic/lag2 contrast '
             'is mostly variable-field drift, not VALUE recovery. Regions in c-node-region-counts '
             'overlap (other_children is also inside descendants_excl_q); use c-unique-damage for '
             'unique arm/graph/node counts. Repeated lag measurements are not independent graphs.',
    )
    write_json(a.out_dir / 'c-audit.json', audit)
    print(json.dumps(dict(out=str(a.out_dir), files=files, denominators=audit['denominators'],
                          checks=checks), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
