"""CPU observation and incremental output, independent of recurrence semantics.

Continuous runners may reuse this on decoded observations, never on hidden
feedback. Importing this module does not import torch or any model code.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

TIMINGS = {'rollout_s': 0.0, 'record_build_s': 0.0}


def configure_runtime():
    """Call before importing torch/model code; read only our CLI option."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--num_threads', type=int, default=4)
    args, _ = parser.parse_known_args()
    if args.num_threads < 1:
        parser.error('--num_threads must be positive')
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'NUMEXPR_NUM_THREADS'):
        os.environ[key] = str(args.num_threads)
    import torch
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)


def add_runtime_arguments(parser):
    parser.add_argument('--num_threads', type=int, default=4)
    parser.add_argument('--progress_every', type=int, default=1,
                        help='print progress every N completed records (minimum 1)')


def cpu_rows(value):
    """One bulk device transfer; scalar indexing happens only on Python lists."""
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    return value


def factor_strings(row, tok, factors=('SYNTAX', 'VARIABLE', 'OPERATION', 'VALUE')):
    row = cpu_rows(row)
    out = {}
    for name in factors:
        value = int(row[tok.factors.index(name)])
        mapping = getattr(tok, name.lower() + '_idx2tok', None)
        out[name] = str(value) if mapping is None else mapping.get(value, str(value))
    return out


class StateRecorder:
    """Cache one graph's fixed truth; copy each observed state just once."""
    def __init__(self, tok, label_b):
        self.tok = tok
        self.labels = cpu_rows(label_b)
        self.truth = {}

    def nodes(self, rows, positions):
        result = {}
        for p in positions:
            if p not in self.truth:
                self.truth[p] = factor_strings(self.labels[p], self.tok)
            true4 = self.truth[p]
            pred4 = factor_strings(rows[p], self.tok)
            result[str(p)] = dict(pos=int(p), pred4=pred4, true4=true4,
                                 first4_correct=pred4 == true4,
                                 value_correct=pred4['VALUE'] == true4['VALUE'])
        return result

    def rounds(self, inputs, outputs, positions, pre_forward_round,
               new_recorder=None, truth_ref=False):
        started = time.perf_counter()
        result = []
        for t in range(len(inputs)):
            inp, out = cpu_rows(inputs[t][0]), cpu_rows(outputs[t][0])
            input_nodes = self.nodes(inp, positions)
            output_nodes = self.nodes(out, positions)
            row = dict(post_round=t + 1, pre_forward_round=pre_forward_round,
                       input={p: n['pred4']['VALUE'] for p, n in input_nodes.items()},
                       output={p: n['pred4']['VALUE'] for p, n in output_nodes.items()},
                       input_nodes=input_nodes, output_nodes=output_nodes)
            if truth_ref:
                row['truth_ref'] = 'old'
            if new_recorder is not None:
                row['input_nodes_new'] = new_recorder.nodes(inp, positions)
                row['output_nodes_new'] = new_recorder.nodes(out, positions)
                row['truth_ref_new'] = 'new'
            result.append(row)
        TIMINGS['record_build_s'] += time.perf_counter() - started
        return result


class RecordWriter:
    """Flush complete records to JSONL; stream final legacy JSON without a list.

The sidecar is a progress snapshot, not a resume/checkpoint mechanism. A killed
process can lose its current record, but completed flushed lines are retained.
"""
    def __init__(self, args, counts, total_graphs):
        self.args = vars(args).copy()
        self.counts = counts
        self.total_graphs = total_graphs
        self.out = Path(args.out)
        self.journal = Path(str(self.out) + '.jsonl')
        self.meta = Path(str(self.out) + '.meta.json')
        self.every = max(1, args.progress_every)
        self.started = time.perf_counter()
        self.written = 0
        self.scanned = 0
        self.out.parent.mkdir(parents=True, exist_ok=True)
        for path in (self.out, self.journal, self.meta):
            if path.exists():
                raise FileExistsError(f'use a fresh output path; already exists: {path}')
        self.file = self.journal.open('x', encoding='utf-8')
        self.progress(force=True)

    def append(self, record):
        self.file.write(json.dumps(record, ensure_ascii=True) + '\n')
        self.file.flush()
        self.written += 1
        self.progress()

    def graph_done(self):
        self.scanned += 1
        self.progress(force=True)

    def progress(self, force=False):
        if not force and self.written % self.every:
            return
        elapsed = time.perf_counter() - self.started
        snapshot = dict(args=self.args, counts=dict(self.counts),
                        written_records=self.written, scanned_graphs=self.scanned,
                        total_graphs=self.total_graphs, elapsed_s=elapsed,
                        timings=dict(TIMINGS), complete=False)
        tmp = Path(str(self.meta) + '.tmp')
        tmp.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')
        os.replace(tmp, self.meta)
        eta = (elapsed / self.scanned * max(0, self.total_graphs - self.scanned)
               if self.scanned else None)
        print(json.dumps(dict(progress=snapshot, records_per_s=self.written / max(elapsed, 1e-9),
                              eta_s=eta)), flush=True)

    def finish(self):
        self.file.close()
        self.progress(force=True)
        write_json(self.journal, self.out, dict(args=self.args, counts=dict(self.counts)))
        snapshot = json.loads(self.meta.read_text(encoding='utf-8'))
        snapshot['complete'] = True
        self.meta.write_text(json.dumps(snapshot, indent=2), encoding='utf-8')


def write_json(journal, out, header, allow_truncated=False):
    """Stream existing records into the original {args, counts, records} format."""
    out = Path(out)
    if out.exists():
        raise FileExistsError(out)
    tmp = Path(str(out) + '.tmp')
    n = 0
    with Path(journal).open(encoding='utf-8') as src, tmp.open('w', encoding='utf-8') as dst:
        dst.write(json.dumps(header)[:-1] + ', "records": [\n')
        for line in src:
            # Only an unterminated last line may be lost during interruption.
            if allow_truncated and not line.endswith('\n'):
                break
            json.loads(line)
            if n:
                dst.write(',\n')
            dst.write(line.rstrip('\n'))
            n += 1
        dst.write('\n]}\n')
    os.replace(tmp, out)
    return n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Recover completed JSONL records as PARTIAL output')
    parser.add_argument('--journal', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    meta_path = args.journal.removesuffix('.jsonl') + '.meta.json'
    meta = json.loads(Path(meta_path).read_text(encoding='utf-8'))
    n = write_json(args.journal, args.out,
                   dict(args=meta['args'], counts=meta['counts'], partial=True,
                        counts_are_progress_snapshot=True), allow_truncated=True)
    print(f'Recovered {n} records; PARTIAL output, counts are a progress snapshot.')
