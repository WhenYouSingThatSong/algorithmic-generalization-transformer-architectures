#!/usr/bin/env python3
"""CPU summary with separate forced/release clocks, doses and clean-paired damage."""
import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import stress_core as S
U, C = S.U, S.C


class Stream:
    def __init__(self, path):
        self.file = path.open('x', newline='', encoding='utf-8')
        self.writer = None

    def write(self, row):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.file.close()


class Totals:
    """Stream counts; fixed replicate count makes mean(trial rates) graph-equal."""
    def __init__(self):
        self.count = 0
        self.sums = defaultdict(float)
        self.ratios = defaultdict(float)
        self.damage_graphs = {}
        self.norm_min = float('inf')
        self.norm_max = 0.

    def add(self, row):
        self.count += 1
        for k, v in row.items():
            if isinstance(v, (int, float)) and k != 'graph_id':
                self.sums[k] += v
        if row.get('n'):
            for k in ('correct', 'harmed', 'excess_wrong'):
                if k in row:
                    self.ratios[k] += row[k]/row['n']
        if row.get('damaged'):
            entry = self.damage_graphs.setdefault(row['graph_id'], [0, 0])
            entry[0] += row['final_two']; entry[1] += row['damaged']
        if 'u_norm' in row:
            self.norm_min = min(self.norm_min, row['u_norm'])
            self.norm_max = max(self.norm_max, row['u_norm'])

    def graphs(self, condition):
        repetitions = S.REPLICATES if condition.startswith('noise_s') else 1
        if self.count % repetitions:
            raise ValueError('missing noise direction')
        return self.count//repetitions

    def ratio(self, k):
        return self.ratios[k]/self.count

    def recovery(self):
        vals = [a/b for a, b in self.damage_graphs.values()]
        return sum(vals)/len(vals) if vals else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(); meta = C.P.read_json(a.run_dir/'meta.json')
    if meta['status'] != 'complete':
        raise ValueError('incomplete run')
    coverage = C.read_cases(a.run_dir/'coverage.jsonl')
    arms = {s['arm'] for s in meta['models']}
    common = S.cohort_keys([r for r in coverage if r['status'] == 'selected'], arms, meta['durations'])
    selected_slots = {(r['arm'], r['graph_id'], r['timing'], r['duration'])
                      for r in coverage if r['status'] == 'selected'}
    record_slots = set()
    for arm in arms:
        if {r['graph_id'] for r in coverage if r['arm'] == arm} != set(range(meta['candidate_graphs'])):
            raise ValueError('coverage omitted fixed candidates')
    a.out_dir.mkdir(parents=True, exist_ok=False)
    streams = {n: Stream(a.out_dir/(n+'.csv')) for n in ('dose', 'trajectory', 'coverage', 'target_events')}
    curves, events, dose_buckets = defaultdict(Totals), defaultdict(Totals), defaultdict(Totals)
    def read_records():
        with (a.run_dir/'records.jsonl').open(encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)
    for r in read_records():
        slot = (r['arm'], r['graph_id'], r['timing'], r['duration'])
        if slot in record_slots:
            raise ValueError('duplicate record')
        record_slots.add(slot)
        if {b['name'] for b in meta['branches']} != set(r['branches']):
            raise ValueError('missing branch/direction')
        key = (r['graph_id'], r['timing']); value, strict = common[key]
        cohorts = ['all_fixed'] + (['common_value'] if value else []) + (['common_strict'] if strict else [])
        pos, gold = r['positions'], r['truth4']; u, q = r['u'], r['q']
        ui, qi = pos.index(u), pos.index(q)
        groups = {'u': [u], 'q': [q], 'other_children': [n for n in r['children'] if n != q],
                  'descendants': r['descendants'], 'unrelated': sorted(set(pos)-{u}-set(r['descendants'])), 'all': pos}
        ref = 'free_q_clean' if r['mode'] == 'sustained' else 'free'
        clean = r['branches'][ref]
        total = r['duration']+meta['horizon'] if r['mode'] == 'sustained' else meta['horizon']
        idbase = {k: r[k] for k in ('arm', 'graph_id', 'timing', 'duration')}
        streams['coverage'].write(dict(**idbase, value_qualified=r['value_qualified'],
            strict_qualified=r['strict_qualified'], common_value=value, common_strict=strict,
            start_graph_value_all=r['start_graph_value_all'], start_graph_first4_all=r['start_graph_first4_all']))
        for name, b in r['branches'].items():
            if len(b['inputs']) != total or len(b['outputs']) != total:
                raise ValueError('wrong observation horizon')
            matched = r['branches']['free_q_wrong'] if b['spec']['q_wrong'] else clean
            ident = dict(**idbase, branch=name, condition=b['spec']['condition'], replicate=b['spec']['replicate'])
            for step, (ip, op, dose) in enumerate(zip(b['inputs'], b['outputs'], b['dose']), 1):
                ph, rel = S.phase(r['mode'], step, r['duration'])
                previous = clean['inputs'][0] if step == 1 else b['outputs'][step-2]
                before = previous[ui][3] == gold[ui][3]
                row = dict(**ident, step=step, phase=ph, phase_round=rel,
                    common_value=value, common_strict=strict, **dose,
                    noise_scale=b['spec'].get('scale'),
                    noise_seed=C.stable_seed(r['noise_seed'], b['spec']['replicate']) if b['spec']['u_mode'] == 'noise' else None,
                    u_before_correct=before, u_input_correct=ip[ui][3] == gold[ui][3],
                    u_readout_changed=ip[ui][3] != previous[ui][3],
                    u_new_value_error=before and ip[ui][3] != gold[ui][3],
                    u_input_expected=ip[ui][3] == r['expected_u'],
                    u_input_nonnumeric=not 0 <= ip[ui][3] < 23,
                    u_input_first4_correct=ip[ui] == gold[ui],
                    u_new_first4_error=previous[ui] == gold[ui] and ip[ui] != gold[ui],
                    u_before_first4_correct=previous[ui] == gold[ui],
                    q_input_correct=ip[qi][3] == gold[qi][3],
                    u_output_correct=op[ui][3] == gold[ui][3],
                    child_n=len(r['children']),
                    children_output_harmed=sum(op[pos.index(n)][3] != gold[pos.index(n)][3]
                        and clean['outputs'][step-1][pos.index(n)][3] == gold[pos.index(n)][3] for n in r['children']))
                # Include only actual intervention opportunities, not release no-ops.
                if S.should_replace(r['mode'], step, r['duration']):
                    streams['dose'].write(row)
                    for cohort in cohorts:
                        dk = (r['arm'], r['timing'], r['duration'], b['spec']['condition'], cohort, step)
                        dose_buckets[dk].add(row)
                streams['trajectory'].write(dict(**ident, step=step, phase=ph, phase_round=rel,
                    absolute_input_state_round=r['state_round']+step-1,
                    u=u, q=q, u_truth=gold[ui][3], q_truth=gold[qi][3],
                    u_input=ip[ui][3], u_output=op[ui][3], q_input=ip[qi][3], q_output=op[qi][3],
                    u_input_parent_sum=U.input_parent_sum(ip, pos, r['parents'], u),
                    q_input_parent_sum=U.input_parent_sum(ip, pos, r['parents'], q),
                    clean_u=clean['outputs'][step-1][ui][3], clean_q=clean['outputs'][step-1][qi][3],
                    u_input_first4=str(ip[ui]), u_output_first4=str(op[ui]),
                    q_input_first4=str(ip[qi]), q_output_first4=str(op[qi])))
            for metric in ('value', 'first4'):
                def ok(row, i):
                    return row[i][3] == gold[i][3] if metric == 'value' else row[i] == gold[i]
                for group, ps in groups.items():
                    if not ps:
                        continue
                    indices = [pos.index(n) for n in ps]
                    node_events = [S.event_windows(r, name, ref, lambda row, i=i: ok(row, i)) for i in indices]
                    if group in ('u', 'q'):
                        for window, ev in node_events[0].items():
                            streams['target_events'].write(dict(**ident, group=group, metric=metric,
                                window=window, common_value=value, common_strict=strict, **ev))
                    for window in node_events[0]:
                        es = [e[window] for e in node_events]; damaged = [e for e in es if e['damaged']]
                        erow = dict(graph_id=r['graph_id'], n=len(ps), damaged=len(damaged),
                            ever_recovered=sum(e['ever_recovered'] for e in damaged),
                            reerror=sum(e['recovered_then_error'] for e in damaged),
                            final_two=sum(e['final_two_correct'] for e in damaged),
                            excess_wrong=sum(e['excess_wrong_rounds'] for e in es))
                        for cohort in cohorts:
                            ek = (r['arm'], r['timing'], r['duration'], b['spec']['condition'], cohort, group, metric, window)
                            events[ek].add(erow)
                    for step, op in enumerate(b['outputs'], 1):
                        ph, rel = S.phase(r['mode'], step, r['duration'])
                        cr, mr = clean['outputs'][step-1], matched['outputs'][step-1]
                        crow = dict(graph_id=r['graph_id'], n=len(ps), correct=sum(ok(op, i) for i in indices),
                            harmed=sum(not ok(op, i) and ok(cr, i) for i in indices),
                            helped=sum(ok(op, i) and not ok(cr, i) for i in indices),
                            matched_q_harmed=sum(not ok(op, i) and ok(mr, i) for i in indices),
                            matched_q_helped=sum(ok(op, i) and not ok(mr, i) for i in indices),
                            nonnumeric=sum(not 0 <= op[i][3] < 23 for i in indices))
                        for cohort in cohorts:
                            ck = (r['arm'], r['timing'], r['duration'], b['spec']['condition'], cohort, group, metric, step, ph, rel)
                            curves[ck].add(crow)
    if selected_slots != record_slots:
        raise ValueError('coverage/records differ')
    for stream in streams.values():
        stream.close()
    fields = ['arm', 'timing', 'duration', 'condition', 'cohort', 'group', 'metric']
    rows = []
    for key, rs in sorted(curves.items()):
        n = int(rs.sums['n'])
        row = dict(zip(fields+['step', 'phase', 'phase_round'], key),
                   graphs=rs.graphs(key[3]), trials=rs.count, n=n,
                   graph_equal=rs.ratio('correct'), node_acc=rs.sums['correct']/n)
        row.update({k: int(rs.sums[k]) for k in ('correct', 'harmed', 'helped', 'matched_q_harmed', 'matched_q_helped', 'nonnumeric')})
        row['harmed_graph_equal'] = rs.ratio('harmed')
        rows.append(row)
    U.write_csv(a.out_dir/'accuracy.csv', rows); rows = []
    for key, rs in sorted(events.items()):
        row = dict(zip(fields+['window'], key), graphs=rs.graphs(key[3]), trials=rs.count,
            observed=int(rs.sums['n']), damaged=int(rs.sums['damaged']),
            ever_recovered=int(rs.sums['ever_recovered']), reerror=int(rs.sums['reerror']),
            final_two_among_damaged=int(rs.sums['final_two']),
            graphs_with_damage=len(rs.damage_graphs), final_two_graph_equal=rs.recovery(),
            excess_wrong_graph_equal=rs.ratio('excess_wrong'))
        rows.append(row)
    U.write_csv(a.out_dir/'events.csv', rows); rows = []
    for key, rs in sorted(dose_buckets.items()):
        row = dict(zip(['arm', 'timing', 'duration', 'condition', 'cohort', 'step'], key),
                   graphs=rs.graphs(key[3]), trials=rs.count)
        for k in ('u_applied', 'q_applied', 'u_before_correct', 'u_input_correct', 'u_readout_changed', 'u_new_value_error',
                  'u_input_expected', 'u_input_nonnumeric', 'u_input_first4_correct',
                  'u_new_first4_error', 'u_before_first4_correct', 'q_input_correct', 'u_output_correct',
                  'child_n', 'children_output_harmed'):
            row[k] = int(rs.sums[k])
        row['new_value_error_rate'] = row['u_new_value_error']/row['u_before_correct'] if row['u_before_correct'] else None
        row['u_norm_mean'] = rs.sums['u_norm']/rs.count
        row['u_norm_min'] = rs.norm_min
        row['u_norm_max'] = rs.norm_max
        rows.append(row)
    U.write_csv(a.out_dir/'dose_summary.csv', rows)
    C.P.write_json(a.out_dir/'meta.json', dict(status='complete', source_meta=meta,
        code_hashes={n: C.P.sha(U.HERE/n) for n in ('summarize_stress.py', 'stress_core.py')},
        records_sha256=C.P.sha(a.run_dir/'records.jsonl'),
        coverage_sha256=C.P.sha(a.run_dir/'coverage.jsonl'),
        common_value_cases=sum(v for v, s in common.values()), common_strict_cases=sum(s for v, s in common.values()),
        structural_skips=[r for r in coverage if r['status'] != 'selected'],
        note='Counts are node/direction trials; directions averaged within graph. Release events start at last forced output. No automatic noise scale selection.'))


if __name__ == '__main__':
    main()
