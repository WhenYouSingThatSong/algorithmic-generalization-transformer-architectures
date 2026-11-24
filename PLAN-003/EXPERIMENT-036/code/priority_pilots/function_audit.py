#!/usr/bin/env python3
"""E/M first step: input-aligned readout/child-output audit of EXISTING C records.
No hidden-state intervention; quadrants are observational candidate evidence.
Supports C_uq_pair_eval_v1 and C_stress_v1 (ADD mod23) only.
"""
import argparse
import sys
from collections import defaultdict
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from io_utils import CSV, jsonl, read_json, sha, write_json

def key(r):
    return r["graph_id"], r["timing"], r.get("duration", 1)

def reference(r, name):
    if r.get("mode") == "sustained":
        return "free_q_wrong" if r["branches"][name]["spec"]["q_wrong"] else "free_q_clean"
    if r.get("mode") == "noise":
        return "free"
    return "q_wrong" if name in ("q_wrong", "both_wrong") else (
        "q_same" if name in ("q_same", "both_same") else "unchanged")

def inputs(r, b):
    if "inputs" in b:
        if len(b["inputs"]) != len(b["outputs"]):
            raise ValueError("input/output length differs")
        return b["inputs"]
    if "mode" in r:
        raise ValueError("stress records MUST contain actual forced inputs")
    return [b["initial"]] + b["outputs"][:-1]

def value(row, positions, p):
    return row[positions.index(p)][3]

def event(flags):
    bad = next((i for i, good in enumerate(flags) if not good), None)
    back = next((i for i in range(bad + 1, len(flags)) if flags[i]), None) if bad is not None else None
    stable = next((i for i in range(bad + 1, len(flags)) if all(flags[i:])), None) if bad is not None else None
    return dict(first_wrong_output=None if bad is None else bad+1,
                first_return_output=None if back is None else back+1,
                stable_return_output=None if stable is None else stable+1,
                reerror_after_return=back is not None and not all(flags[back:]))

def edge_rows(r, name):
    pos, gold = r["positions"], r["truth4"]
    if len(set(pos)) != len(pos) or len(pos) != len(gold):
        raise ValueError("invalid node identities")
    if any(not isinstance(g[3], int) or not 0 <= g[3] < 23 for g in gold):
        raise ValueError("expected numeric mod23 VALUE IDs 0..22")
    u = r["u"]; b = r["branches"][name]
    refname = reference(r, name)
    ref = r["branches"][refname]
    ip = inputs(r, b)
    if len(ref["outputs"]) != len(b["outputs"]):
        raise ValueError("reference horizon differs")
    qs = sorted(set(r["children"]))
    for q in qs:
        parents = r["parents"].get(str(q), r["parents"].get(q))
        if not parents or u not in parents or any(p not in pos for p in parents):
            raise ValueError("unsupported child expression")
        if sum(gold[pos.index(p)][3] for p in parents)%23 != gold[pos.index(q)][3]:
            raise ValueError("source truth does not satisfy ADD mod23")
        for i, out in enumerate(b["outputs"]):
            inp = ip[i]; clean = ref["outputs"][i]
            if any(len(x) != len(pos) for x in (inp, out, clean)):
                raise ValueError("state/position length differs")
            vs = [value(inp, pos, p) for p in parents]
            all_numeric = all(isinstance(v, int) and 0 <= v < 23 for v in vs)
            qi, ui = pos.index(q), pos.index(u)
            step = i + 1; dur = r.get("duration", 1)
            forced = r.get("mode") == "sustained" and step <= dur
            yield dict(graph_id=r["graph_id"], timing=r["timing"], duration=dur,
                branch=name, reference=refname, post_round=step,
                phase="forced" if forced else ("released" if r.get("mode") == "sustained" else "post"),
                release_round=step-dur if r.get("mode") == "sustained" and not forced else None,
                absolute_output_round=r["state_round"]+step, u=u, q=q,
                focal_q=q == r["q"], u_input=value(inp,pos,u), u_output=value(out,pos,u),
                q_input=value(inp,pos,q), q_output=value(out,pos,q),
                u_input_correct=value(inp,pos,u)==gold[ui][3],
                u_output_correct=value(out,pos,u)==gold[ui][3],
                u_input_first4_correct=inp[ui]==gold[ui],
                q_input_correct=value(inp,pos,q)==gold[qi][3],
                q_output_correct=value(out,pos,q)==gold[qi][3],
                q_output_first4_correct=out[qi]==gold[qi],
                q_output_numeric=isinstance(value(out,pos,q),int) and 0<=value(out,pos,q)<23,
                q_reference_correct=value(clean,pos,q)==gold[qi][3],
                other_parents_input_correct=all(value(inp,pos,p)==gold[pos.index(p)][3] for p in parents if p != u),
                other_parents_input_first4_correct=all(inp[pos.index(p)]==gold[pos.index(p)] for p in parents if p != u),
                all_parents_input_correct=all(value(inp,pos,p)==gold[pos.index(p)][3] for p in parents),
                input_parent_sum=sum(vs)%23 if all_numeric else None,
                matches_input_parent_sum=value(out,pos,q)==sum(vs)%23 if all_numeric else None,
                extra_q_damage=value(clean,pos,q)==gold[qi][3] and value(out,pos,q)!=gold[qi][3])

def run(sources, out):
    """Keep independent source IDs, models, timings, durations, and every round."""
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    meta = dict(status="running", protocol="C_function_audit_v1",
                note="Readout/child-output observations, NOT proof of internal use or recovery.",
                sources={}, code={Path(__file__).name:sha(__file__), "io_utils.py":sha(Path(__file__).with_name("io_utils.py"))})
    write_json(out/"meta.json", meta)
    rows_csv, events_csv, coverage_csv = [CSV(out/n) for n in ("edge_rounds.csv","events.csv","coverage.csv")]
    buckets = defaultdict(lambda: defaultdict(lambda: [0,0,0,0]))
    nrows = 0
    try:
        for source, path in sources.items():
            path=Path(path); sm=read_json(path/"meta.json")
            if sm.get("status")!="complete" or sm.get("protocol") not in ("C_uq_pair_eval_v1","C_stress_v1"):
                raise ValueError("unsupported or incomplete source: "+source)
            arms={x["arm"] for x in sm["models"]}
            durations=set(sm.get("durations",[1]))
            eligibility={}; seen=set()
            for r in jsonl(path/"records.jsonl"):
                identity=(r["arm"],*key(r))
                if identity in seen: raise ValueError("duplicate source record")
                seen.add(identity)
                pair=(r["graph_id"],r["timing"])
                eligibility.setdefault(pair,{})[(r["arm"],r.get("duration",1))]=bool(r["value_qualified"])
            expected={(arm,d) for arm in arms for d in durations}
            if any(set(v)!=expected for v in eligibility.values()):
                raise ValueError("models/durations not paired within source")
            common={k:all(v.values()) for k,v in eligibility.items()}
            meta["sources"][source]=dict(path=str(path),meta_sha256=sha(path/"meta.json"),
                records_sha256=sha(path/"records.jsonl"),source_meta=sm,
                coverage_sha256=sha(path/"coverage.jsonl") if (path/"coverage.jsonl").exists() else None,
                records=len(seen),paired_graph_timings=len(common),
                common_graph_timings=sum(common.values()))
            for r in jsonl(path/"records.jsonl"):
                iscommon=common[(r["graph_id"],r["timing"])]
                coverage_csv.add(dict(source=source,arm=r["arm"],graph_id=r["graph_id"],
                    timing=r["timing"],duration=r.get("duration",1),own_value_qualified=r["value_qualified"],
                    common_value_qualified=iscommon))
                for name in r["branches"]:
                    er=list(edge_rows(r,name)); byq=defaultdict(list)
                    for row in er:
                        row=dict(source=source,arm=r["arm"],common_value_qualified=iscommon,**row)
                        rows_csv.add(row); nrows+=1; byq[row["q"]].append(row)
                        contexts=["all"]
                        if row["other_parents_input_correct"] and row["q_reference_correct"]:
                            contexts += ["other_parents_and_reference_correct",
                                "controlled_q_input_correct" if row["q_input_correct"] else "controlled_q_input_wrong"]
                        if row["other_parents_input_first4_correct"] and row["q_reference_correct"]:
                            contexts.append("other_parents_first4_and_reference_correct")
                        for cohort in ["all_recorded"]+(["common_value"] if iscommon else []):
                            for context in contexts:
                                for axis in ("u_input_correct","u_output_correct"):
                                    bk=(source,r["arm"],r["timing"],r.get("duration",1),name,
                                        cohort,row["post_round"],axis,context)
                                    cell=(2 if row[axis] else 0)+(1 if row["q_output_correct"] else 0)
                                    buckets[bk][r["graph_id"]][cell]+=1
                    for q, qr in byq.items():
                        events_csv.add(dict(source=source,arm=r["arm"],graph_id=r["graph_id"],
                            timing=r["timing"],duration=r.get("duration",1),branch=name,q=q,
                            common_value_qualified=iscommon,
                            target_read_correct_child_wrong_rounds=sum(x["u_output_correct"] and not x["q_output_correct"] for x in qr),
                            **event([x["q_output_correct"] for x in qr])))
        summary=CSV(out/"quadrants.csv")
        try:
            for bk, gs in sorted(buckets.items()):
                total=sum(sum(v) for v in gs.values())
                for cell in range(4):
                    summary.add(dict(zip(("source","arm","timing","duration","branch","cohort","post_round","axis","context"),bk),
                        u_read_correct=bool(cell//2),q_output_correct=bool(cell%2),
                        graph_n=len(gs),edge_n=total,cell_n=sum(v[cell] for v in gs.values()),
                        micro_fraction=sum(v[cell] for v in gs.values())/total,
                        graph_macro_fraction=sum(v[cell]/sum(v) for v in gs.values())/len(gs)))
        finally: summary.close()
        meta.update(status="complete",edge_round_rows=nrows)
        write_json(out/"meta.json",meta)
    finally:
        for f in (rows_csv,events_csv,coverage_csv):f.close()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",action="append",required=True,help="unique_name=existing_run_directory")
    p.add_argument("--out-dir",type=Path,required=True)
    a=p.parse_args(); sources={}
    for item in a.source:
        name,path=item.split("=",1)
        if name in sources: p.error("duplicate source name")
        sources[name]=path
    run(sources,a.out_dir)
if __name__=="__main__":main()
