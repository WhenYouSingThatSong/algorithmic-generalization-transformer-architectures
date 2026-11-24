#!/usr/bin/env python3
"""W1/W2: paired original-task corruption vs legitimate constant reassignment.
Discrete A'/B only. No training, no slot recycling, no oracle answers in inputs.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from io_utils import read_json, write_json, sha
import reassignment_core as P

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]

def encode_rows(rows, tok, positions):
    maps=[getattr(tok,f.lower()+"_idx2tok") for f in tok.factors[:4]]
    return [[str(maps[i][int(rows[p][i])]) for i in range(4)] for p in positions]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest",type=Path,required=True)
    p.add_argument("--model",choices=("model_a","model_b"),required=True)
    p.add_argument("--out-dir",type=Path,required=True)
    p.add_argument("--candidate_start",type=int,default=32)
    p.add_argument("--candidate_count",type=int,default=32)
    p.add_argument("--horizon",type=int,default=16)
    p.add_argument("--batch_size",type=int,default=8)
    p.add_argument("--num_threads",type=int,default=4)
    a=p.parse_args()
    if min(a.candidate_count,a.horizon,a.batch_size,a.num_threads)<1 or a.candidate_start<0:
        p.error("positive budgets and nonnegative start required")
    sys.path.insert(0,str(HERE.parent/"b_mechanism"))
    from trajectory_io import configure_runtime
    configure_runtime()
    import torch
    import b_mechanism_common as C
    manifest=read_json(a.manifest); spec=manifest[a.model]
    ckpt=Path(C.checkpoint_path(spec["ckpt_base_dir"],spec["group_name"],spec["run_name"],spec["ckpt_name"]))
    ds=Path(manifest["ds_file"])
    C.verify_manifest_identity(manifest,str(ds),str(ckpt)); C.validate_manifest_graphs(manifest)
    if sha(ckpt)!=spec["ckpt_sha256"]:raise ValueError("wrong requested model")
    gids=set(range(a.candidate_start,a.candidate_start+a.candidate_count))
    mg={g["graph_id"]:g for g in manifest["graphs"] if g["graph_id"] in gids}
    a.out_dir.mkdir(parents=True,exist_ok=False)
    deps=[Path(__file__),HERE/"reassignment_core.py",HERE/"io_utils.py",
          HERE.parent/"b_mechanism/b_mechanism_common.py",ROOT/"experiments/model.py",
          ROOT/"experiments/tokenizers.py"]
    meta=dict(status="running",protocol="discrete_reassignment_v1",model_key=a.model,model=spec,
        dataset=str(ds),dataset_sha256=sha(ds),manifest_sha256=sha(a.manifest),
        candidate_ids=sorted(gids),horizon=a.horizon,batch_size=a.batch_size,deltas=P.DELTAS,
        num_threads=a.num_threads,code_hashes={str(x.relative_to(ROOT)):sha(x) for x in deps},
        note="Used N32 development graphs; node values/structure must match old truth before intervention. No new-node capacity claim.")
    write_json(a.out_dir/"meta.json",meta)
    ctx=C.load_eval_context(spec["config_dir_name"],spec["group_name"],spec["run_name"],
        spec["ckpt_base_dir"],spec["ckpt_name"],str(ds),batch_size=a.batch_size,num_workers=0)
    lit,tok,teacher,mc=(ctx[k] for k in ("lit","tok","teacher","mc"))
    if lit.nointerm or tok.mod_val!=23:raise ValueError("this pilot requires discrete mod23")
    meta["runtime"]=dict(python=sys.version,torch=torch.__version__,device=str(ctx["device"]))
    write_json(a.out_dir/"meta.json",meta)
    seen=set(); selected=0; records=0
    with (a.out_dir/"selection.jsonl").open("x",encoding="utf-8") as cov, \
         (a.out_dir/"records.jsonl").open("x",encoding="utf-8") as dst,torch.no_grad():
        for batch_i, raw_batch in enumerate(ctx["dm"].val_dataloader()):
            if batch_i*a.batch_size>max(gids):break
            data,label,info=C.split_batch(lit.reduce_batch(raw_batch).to(ctx["device"]),mc)
            for bi in range(len(data)):
                gid=batch_i*a.batch_size+bi
                if gid not in gids:continue
                seen.add(gid)
                g=mg.get(gid)
                if g is None or g.get("T") is None:
                    cov.write(json.dumps(dict(graph_id=gid,status="no_common_manifest_completion"))+"\n");continue
                graph=C.build_meta(data[bi],label[bi],info[bi],tok,teacher)
                nodes=graph["nodes"]; depths={q:int(graph["depth"][q]) for q in nodes}
                parents=graph["parents"]
                raw=data[bi].cpu().tolist(); labs=label[bi].cpu().tolist()
                for q in nodes:
                    ps,ops=C.parse_expr(info[bi],q)
                    if len(ops)!=len(ps)-1 or any(tok.operation_idx2tok[o]!="ADD" for o in ops):
                        raise ValueError("non-ADD or malformed expression")
                constant_positions=sorted({p for ps in parents.values() for p in ps if p not in nodes})
                if any(tok.syntax_idx2tok[raw[p][C.S]]!="VALUE" for p in constant_positions):
                    raise ValueError("expression references a nonconstant/nonnodal position")
                constants={p:int(C.tok_str(tok,raw[p][C.V])) for p in constant_positions}
                old={q:int(C.tok_str(tok,labs[q][C.V])) for q in nodes}
                choice=P.select_leaf(nodes,depths,parents,constants,gid)
                if choice is None:
                    cov.write(json.dumps(dict(graph_id=gid,status="no_leaf_with_descendant"))+"\n");continue
                leaf,const=choice
                _,outs=C.rollout(lit,teacher,data[bi:bi+1],g["T"]+1)
                base=outs[-1]
                valid=graph["valid"]
                value_ok=C.all_numeric_correct(base[0],label[bi],graph["numeric"])
                strict_ok=bool((base[0,valid,:4]==label[bi,valid,:4]).all())
                eligibility=dict(graph_id=gid,leaf=leaf,constant=const,state_round=g["T"]+1,
                    value_qualified=value_ok,strict_qualified=strict_ok,
                    status="selected" if strict_ok and value_ok else "old_state_not_strictly_correct")
                cov.write(json.dumps(eligibility)+"\n");cov.flush()
                if not strict_ok or not value_ok:continue
                selected+=1
                base_rows=base[0].cpu().tolist()
                positions=sorted(set(nodes+[const]))
                old4=encode_rows(labs,tok,positions)
                for delta in P.DELTAS:
                    plan=P.make_plan(nodes,depths,parents,constants,old,leaf,const,delta)
                    starts=P.starts(base_rows,raw,plan,C.V,lambda v:tok.value_tok2idx[str(v)])
                    xs=torch.tensor([starts[n]["state"] for n in P.BRANCHES],device=ctx["device"],dtype=data.dtype)
                    C.force_aux_false(xs,teacher)
                    new_id=tok.value_tok2idx[str(plan["new_constant"])]
                    def hook(step,cur):
                        for j,n in enumerate(P.BRANCHES):
                            if starts[n]["pinned"]:cur[j,const,C.V]=new_id
                        return cur
                    ins,os=C.rollout(lit,teacher,xs,a.horizon,input_hook=hook)
                    ips=[x.cpu().tolist() for x in ins]; ops=[x.cpu().tolist() for x in os]
                    branches={}
                    for j,n in enumerate(P.BRANCHES):
                        branches[n]=dict(truth_ref=starts[n]["truth_ref"],start_kind=starts[n]["start_kind"],
                            pre_forward_round=0 if starts[n]["start_kind"]=="scratch" else g["T"]+1,
                            pinned=starts[n]["pinned"],initial=encode_rows(starts[n]["state"],tok,positions),
                            inputs=[encode_rows(x[j],tok,positions) for x in ips],
                            outputs=[encode_rows(x[j],tok,positions) for x in ops])
                    if branches["clean"]!=branches["sham"]:
                        raise ValueError("exact-writeback sham differs")
                    for n in P.BRANCHES:
                        if starts[n]["pinned"] and any(x[positions.index(const)][3]!=str(plan["new_constant"]) for x in branches[n]["inputs"]):
                            raise ValueError("new constant not actually pinned")
                    new4=[list(v) for v in old4]
                    for i,pos in enumerate(positions):new4[i][3]=str(plan["new_truth"][pos])
                    rec=dict(graph_id=gid,state_round=g["T"]+1,leaf=leaf,constant=const,delta=delta,
                        positions=positions,old_truth4=old4,new_truth4=new4,regions=plan["regions"],
                        nodes=nodes,depths=depths,parents=parents,constants=constants,
                        new_constant=plan["new_constant"],horizon=a.horizon,
                        raw_input_tokens=raw,completed_tokens=base_rows,branches=branches)
                    dst.write(json.dumps(rec)+"\n");dst.flush();records+=1
                print(json.dumps(dict(model=a.model,graph_id=gid,selected=selected,records=records)),flush=True)
        if seen!=gids:
            raise ValueError("dataset did not cover fixed candidate IDs; do not refill")
    meta.update(status="complete",scanned=len(seen),selected=selected,records=records,
        records_sha256=sha(a.out_dir/"records.jsonl"),selection_sha256=sha(a.out_dir/"selection.jsonl"))
    write_json(a.out_dir/"meta.json",meta)

if __name__=="__main__":main()
