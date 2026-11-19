#!/usr/bin/env python3
"""Build a compact review packet from completed summaries; no new model work."""
import argparse
import csv
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'priority_pilots'))
from io_utils import read_json,write_json,sha,jsonl


def choose_row(r,continuous):
    cond=json.loads(r['condition'])
    if continuous:
        return (cond[2]==2 and r['region']=='q' and r['branch'] in
                ('observed_keep','observed_pre','restore_u_pre','clean_pre','observed_solved') and
                r['cohort'] in ('all_recorded','common_input_qualified') and r['round_or_window']=='last_two')
    return (cond[1]==16 and r['region'] in ('target','changed','all_nodes') and
            r['branch'] in ('warm','clear','scratch','warm_pinned') and
            r['cohort']=='common_source_qualified' and r['round_or_window']=='last_two')


def compact_example(r,continuous):
    if continuous:
        at=r['positions'].index(r['q']);truth=r['truth4'][at][3]
        names=('observed_keep','observed_pre','restore_u_pre','clean_pre','observed_solved')
        good={n:all(v[at][3]==truth for v in r['branches'][n]['outputs'][-2:]) for n in names}
        kind=f"pre={int(good['observed_pre'])},restore_pre={int(good['restore_u_pre'])},clean_pre={int(good['clean_pre'])}"
        nodes=sorted({r['u'],r['q'],*r['parents'].get(str(r['q']),r['parents'].get(r['q'],[]))})
        header=dict(graph_id=r['graph_id'],arm=r['arm'],timing=r['timing'],mode=r['mode'],lag=r['probe_lag'],
            context=r['context'],u=r['u'],q=r['q'],truth4={p:r['truth4'][r['positions'].index(p)] for p in nodes})
    else:
        target=r['plan']['target'];at=r['positions'].index(target);truth=r['new_truth4'][at][3]
        names=('warm','clear','scratch','warm_pinned')
        good={n:all(v[at][3]==truth for v in r['branches'][n]['outputs'][-2:]) for n in names}
        kind=f"warm={int(good['warm'])},clear={int(good['clear'])},scratch={int(good['scratch'])}"
        ps=r['plan']['parents'];nodes=sorted({target,*ps.get(str(target),ps.get(target,[]))})
        header=dict(graph_id=r['graph_id'],schedule=r['schedule'],episode=r['episode'],horizon=r['horizon'],
            plan=r['plan'],truth4={p:r['new_truth4'][r['positions'].index(p)] for p in nodes})
    trace={}
    for name in names:
        b=r['branches'][name]
        trace[name]=[dict(post_round=s+1,
            inputs={p:b['inputs'][s][r['positions'].index(p)] for p in nodes},
            outputs={p:b['outputs'][s][r['positions'].index(p)] for p in nodes}) for s in range(len(b['outputs']))]
    return kind,dict(**header,pattern=kind,last_two_target_correct=good,trace=trace)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--summary-dir',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True)
    a=p.parse_args();meta=read_json(a.summary_dir/'meta.json')
    if meta.get('status')!='complete' or meta.get('protocol')!='state_reuse_summary_v1':raise ValueError('complete summary required')
    continuous=meta['sources'][0]['meta']['protocol']=='C_function_probe_v1'
    with (a.summary_dir/'summary.csv').open(encoding='utf-8-sig',newline='') as f:
        chosen=[r for r in csv.DictReader(f) if choose_row(r,continuous)]
    a.out_dir.mkdir(parents=True,exist_ok=False)
    columns=['model','condition','branch','region','cohort','round_or_window','selected_graph_n','nonempty_graph_n',
             'node_n','value_n','first4_n','value_macro','first4_macro']
    with (a.out_dir/'key-results.csv').open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(chosen)
    # Two smallest graph IDs per observed pattern/model. These are illustrations,
    # explicitly outcome-conditioned; NEVER used as performance denominators.
    examples={}
    for source in meta['sources']:
        path=Path(source['path']);sm=read_json(path/'meta.json')
        if sha(path/'meta.json')!=source['meta_sha256'] or sha(path/'records.jsonl')!=sm['records_sha256']:
            raise ValueError('source records changed since summary')
        for r in jsonl(path/'records.jsonl'):
            if continuous and (r['timing'],r['mode'],r['probe_lag'])!=('late','semantic',2):continue
            if not continuous and r['horizon']!=16:continue
            kind,ex=compact_example(r,continuous)
            arm=r['arm'] if continuous else sm['model_key'];ex['arm']=arm
            k=(arm,kind);pool=examples.setdefault(k,{})
            g=r['graph_id']
            # earliest episode/record for a graph-pattern; then smallest two graphs
            if g not in pool:pool[g]=ex
            for drop in sorted(pool)[2:]:del pool[drop]
    with (a.out_dir/'examples.jsonl').open('x',encoding='utf-8') as f:
        for k,pool in sorted(examples.items()):
            for gid,ex in sorted(pool.items()):f.write(json.dumps(ex,ensure_ascii=False)+'\n')
    lines=['# 本批结果审阅入口','',
        '本表为自动计数摘要，不包含机制判断。完整条件和每轮结果见源summary及records。',
        '节点VALUE与四属性按各图比例再等权平均；末两轮要求两轮均正确。空分母不记100%。',
        'C四版本同源，不是四种子；共同输入资格的覆盖必须与全体并列。',
        'examples按观察到的结果类别选最小两个graph_id，仅作解释例子，不是独立测试集。','',
        '| 模型 | 条件 | 分支 | 区域/分母 | VALUE | 四属性 | 图均权VALUE |',
        '|---|---|---|---|---:|---:|---:|']
    for r in chosen:
        if continuous and (json.loads(r['condition'])!=['late','semantic',2] or r['cohort']!='all_recorded'):continue
        if not continuous and r['region']!='target':continue
        lines.append(f"| {r['model']} | {r['condition']} | {r['branch']} | {r['region']}; {r['node_n']}节点/{r['nonempty_graph_n']}图 | {r['value_n']} | {r['first4_n']} | {r['value_macro']} |")
    if not chosen:lines+=['','未产生主表行：检查候选覆盖、预算和共同资格；不能视作100%或管线失败。']
    lines+=['','交付时补充：真实小验证检查记录、运行版本和预算、无资格/零事件、固定案例说明、支持与未区分的解释。']
    (a.out_dir/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write_json(a.out_dir/'meta.json',dict(status='complete',protocol='state_reuse_delivery_v1',
        source_summary=str(a.summary_dir),summary_meta_sha256=sha(a.summary_dir/'meta.json'),
        summary_csv_sha256=sha(a.summary_dir/'summary.csv'),script_sha256=sha(__file__),rows=len(chosen),
        files={p.name:sha(p) for p in a.out_dir.iterdir() if p.is_file()},
        note='No new experiments. Illustrations conditioned on outcomes, not a prevalence sample.'))


if __name__=='__main__':main()
