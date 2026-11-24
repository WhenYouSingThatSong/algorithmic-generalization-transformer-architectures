"""Does a decoded wrong vector actually mislead the original, untrained-for-correction C?"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import protocol as P

def run_probe(donor_rt,plan,split,limit=64):
    from runtime import Runtime
    from evaluate import load_split,source,baseline,branch
    from collections import defaultdict
    import sys
    sys.path.insert(0,str(P.ROOT/'tools/counterfactual_control'))
    from design import matched_changes
    spec=next(m for m in plan['models'] if m['id']==f"C_s{donor_rt.spec['seed']}")
    if spec['family']!='C':raise ValueError('probe receiver must be original C')
    # Cache original C, never optimizer parameters and never train it.
    if not hasattr(donor_rt,'frozen_receiver'):donor_rt.frozen_receiver=Runtime(spec,str(donor_rt.device))
    receiver=donor_rt.frozen_receiver;receiver.model.eval()
    x,y,gs=load_split(receiver,plan['splits'][split]);records=[]
    counts=defaultdict(lambda:defaultdict(int))
    with receiver.torch.no_grad():
        for gid,g in enumerate(gs[:limit]):
            target=plan['splits'][split]['targets'][gid]
            if target is None:
                records.append(dict(gid=gid,status='no_target'));continue
            u,q,delta=target['u'],target['q'],target['delta'];xx=x[gid:gid+1].to(receiver.device);yy=y[gid].to(receiver.device)
            matched=matched_changes(g['coefs'][u],(delta,))
            if matched:
                changes={name:{g['constants'][p]:v for p,v in edits.items()} for name,edits in matched.items()}
                wrong_edit=changes[f'shift_{delta}'];same_edit=changes['same']
            else:wrong_edit=P.donor_edits(g,u,delta);same_edit=P.same_edits(g,u)
            # Early test uses precisely the source age used for training injection.
            # Recipient has just computed u, so q's other parents can be available.
            for timing,t in (('early',g['depths'][u]+1),('complete_budget',g['D']+4)):
                clean,snaps=baseline(receiver,xx,yy,g,t+4,{t});base=snaps[t]
                start=receiver.decode(base)[0];qidx=g['nodes'].index(q)
                other=[p for p in g['parents'][q] if p!=u]
                eligible=bool((start[u,:4]==yy[u,:4]).all()) and all(bool((start[p,:4]==yy[p,:4]).all()) for p in other)
                eligible=eligible and clean[t]['output'][qidx][0]==int(yy[q,3])
                expected=(sum((int(yy[u,3])+delta)%23 if p==u else int(start[p,3]) for p in g['parents'][q]))%23 if all(0<=int(start[p,3])<23 for p in other) else None
                for origin,rt in (('original_C',receiver),('current_C',donor_rt)):
                    for condition in ('wrong_value','matched_wrong_value','same_value','source_original','receiver_exact'):
                        # wrong_value uses the EXACT one-leaf training construction.
                        # matched_wrong_value is a separate edit-matched comparison.
                        edit=P.donor_edits(g,u,delta) if condition=='wrong_value' else wrong_edit if condition=='matched_wrong_value' else same_edit if condition=='same_value' else {}
                        if condition=='receiver_exact':vec={u:base[0,u].clone()};read={str(u):start[u,:4].tolist()};accepted=True
                        else:vec,read,accepted=source(rt,xx,yy,g,[u],edit,t)
                        label=('same_value_original_source' if condition=='same_value' else 'unmatched_wrong_value') if not matched and condition in ('same_value','matched_wrong_value') else condition
                        row=dict(gid=gid,timing=timing,start=t,origin=origin,condition=label,u=u,q=q,delta=delta,
                                 eligible=eligible,source_read=read,expected_wrong=expected,correct=int(yy[q,3]),
                                 different_source=bool(edit),matched_edits=bool(matched) and condition in ('matched_wrong_value','same_value'),status='ok' if accepted else 'source_rejected')
                        c=counts[(timing,origin,label)];c['selected']+=1;c['source_accepted']+=int(accepted)
                        if not accepted:records.append(row);continue
                        out=branch(receiver,base,g,yy,[u],vec,horizon=4)
                        received=out[0]['input'][g['nodes'].index(u)]
                        intended=int(yy[u,3])
                        if condition in ('wrong_value','matched_wrong_value'):intended=(intended+delta)%23
                        intended_row=yy[u,:4].tolist();intended_row[3]=intended
                        receiver_reads_intended=received==intended_row
                        qvals=[s['output'][qidx][0] for s in out]
                        wrong=qvals[0]!=int(yy[q,3]);specific=expected is not None and expected!=int(yy[q,3]) and qvals[0]==expected
                        row.update(receiver_injected_readout=received,receiver_reads_intended=receiver_reads_intended,
                                   q_values=qvals,first_wrong=wrong,first_expected_wrong=specific)
                        c['eligible_accepted']+=int(eligible);c['first_wrong']+=int(eligible and wrong);c['first_expected_wrong']+=int(eligible and specific)
                        c['receiver_intended']+=int(eligible and receiver_reads_intended)
                        c['first_expected_wrong_given_receiver_intended']+=int(eligible and receiver_reads_intended and specific)
                        records.append(row)
    return dict(receiver=spec,split=split,selected_graphs=min(limit,len(gs)),records=records,
                counts=[dict(timing=k[0],origin=k[1],condition=k[2],**v) for k,v in sorted(counts.items())],
                interpretation='Original-C donors test semantic corruption within its own representation. Current-C donors test actual trained vectors in frozen original C; cross-weight representation shift is a confound, compare same-value/source-original controls. Do not treat no propagation as proof of no error.')

def main():
    import argparse
    from pathlib import Path
    from runtime import Runtime
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--model',required=True);p.add_argument('--split',default='dev');p.add_argument('--limit',type=int,default=64)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda');p.add_argument('--num_threads',type=int,default=4)
    a=p.parse_args();plan=P.read(a.plan);rt=Runtime(next(z for z in plan['models'] if z['id']==a.model),a.device)
    report=run_probe(rt,plan,a.split,a.limit)
    report.update(plan_sha256=P.sha(a.plan),code_hashes=P.source_hashes())
    P.write(a.output,report)

if __name__=='__main__':main()
