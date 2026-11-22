"""Read real per-graph trajectories and write the paired, scenario-by-scenario report."""
import argparse
from collections import Counter,defaultdict
import hashlib
from pathlib import Path

from abc_scene_queue import OUT,C_RAW,DISCRETE,MODELS,P,PROVENANCE,archive,saved
from attribution_queue import csv_output

ORDER=("A_s42","B_s42","C_s42","C_s1234")
NAMES=("离散 A′","离散 B","原始 C（42）","原始 C（1234）")
CN={"first_correct":"目标首次算对","child_correct":"目标与子节点首次均对","graph_complete":"整图首次算完",
    "early":"至少一个直接父值尚未就绪","before":"目标按图深度将要计算","after":"目标按图深度刚算完","complete":"整图标准轮数后再算四轮",
    "internal":"内部节点","source":"输入节点","sink":"输出终点","target":"只改所选节点",
    "parent":"只改直接父节点","both":"目标与父值分别改成冲突值","held_parent":"直接父值持续错四轮",
    "held_same_parent":"父节点持续四轮同值替换","same_value":"同值替换","clean":"干净对照",
    "input_source":"改输入节点已算出的值","direct_parent":"改直接父节点","older_ancestor":"改更早的祖先",
    "separate_branch":"改另一条分支","both_branches":"同时改两条分支",
    "both_parents_add":"两个父值的改动相加","both_parents_cancel":"两个父值的改动相消",
    "parent_and_target_consistent":"父节点与目标按同一改值方案写入",
    "parent_and_target_conflict":"父节点与目标按不同改值方案写入",
    "restore_empty":"将内部节点换回先前的空状态","fill_correct":"给将要计算的位置提前填对",
    "target_and_child":"同时写入内部节点与子节点"}
LEGACY={"target_just_self":"内部节点原样写回","parent_just_self":"父节点原样写回",
    "ancestor_just_self":"祖先原样写回","target_pending_correct":"目标提前填对",
    "parent_pending_correct":"父节点从下一轮正确状态写回","ancestor_pending_correct":"祖先从下一轮正确状态写回",
    "target_pending_empty":"向空目标换入空状态","target_pending_wrong":"目标提前填错",
    "target_just":"内部节点刚算对后改错","parent_pending":"将要计算位置的父值改错",
    "ancestor_pending":"将要计算位置的祖先改错","parent_just":"内部节点刚算对后改父值",
    "ancestor_just":"内部节点刚算对后改祖先","all_pending_wrong":"同时改将算目标、父节点及祖先",
    "all_just":"同时改已算目标、父节点及祖先","later_target":"改已完成图的内部节点",
    "later_parent":"改已完成图的父节点","later_ancestor":"改已完成图的祖先",
    "later_all":"改已完成图的连续三层"}


def grid_observations(case,archived=False):
    nodes=case["nodes"]
    u,q=case["u"],case["q"]
    ys=[n["truth"] for n in nodes]
    clean={e["start"]:e for e in case["events"] if "clean" in e["label"]}
    def values(e):
        if archived:
            return [[z["values"][str(i)] for i in range(len(nodes))] for z in e["rows"]]
        return e["rows"]
    clean_rows={t:values(e) for t,e in clean.items()}
    for e in case["events"]:
        t=e["start"]
        rows=values(e)
        reference=clean_rows[t]
        if archived:
            phase="before" if t<case["due"] else "after" if t==case["due"] else "complete"
            eligible=e["base_ok"]["computed_ok"] and e["base_ok"]["pending_empty"]
            for i,src,_,*rest in e["patches"]:
                want=24 if nodes[i]["depth"]>=src else nodes[i]["truth" if rest==["base"] else "alt"]
                eligible=eligible and e["pre"][str(i)]==want
        else:
            phase=e["phase"]
            eligible=e["value_eligible"]
        damaged=[s for s,(z,c) in enumerate(zip(rows,reference)) if c[q]==ys[q] and z[q]!=ys[q]]
        yield ("chain",phase,e["label"]),dict(t=t,eligible=eligible,
            next_correct=rows[0][u]==ys[u],child_damage=bool(damaged),
            follow=bool(damaged) and damaged[0]<len(rows)-2,
            end_joint=bool(damaged) and damaged[0]<len(rows)-2 and
                      all(z[u]==ys[u] and z[q]==ys[q] for z in rows[-2:]),
            q_risk=any(z[q]==ys[q] for z in reference),
            extra_wrong=sum(a!=y==b for a,b,y in zip(rows[-1],reference[-1],ys)),
            early_numerical_error=any(c[q]==24 and 0<=z[q]<23 and z[q]!=ys[q] for z,c in zip(rows,reference)))


def standard_observations(case):
    for e in case["events"]:
        key=e["role"],e["timing"],e["condition"]
        if "skip" in e:
            yield key,dict(eligible=False,t=e["t"])
            continue
        q=e["gold_child"]
        harm=[s for s,(z,c) in enumerate(zip(e["trace"],e["clean"]))
              if q is not None and c["q"]==q and z["q"]!=q]
        follow=bool(harm) and harm[0]<len(e["trace"])-2
        yield key,dict(t=e["t"],eligible=True,next_correct=e["target_correct_next"],
                       child_damage=bool(harm),q_risk=e["q_at_risk"],follow=follow,
                       end_joint=follow and all(z["u"]==e["gold_target"] and z["q"]==q for z in e["trace"][-2:]),
                       specific=e["specific_child"],specific_follow=e["specific_with_followup"],
                       specific_end=e["joint_stable_after_specific"],
                       retained_injection=e["target_ends_injected"],follows_parent=e["first_uses_wrong_parent"])


def aggregate(root):
    raw={}
    graphs={}
    denom=defaultdict(Counter)
    evidence=defaultdict(Counter)
    fit,natural=[],[]
    hashes=P.read(PROVENANCE)["models"]
    for path in (C_RAW/"raw-manifest.sha256").read_text().splitlines():
        digest,filename=path.split()
        path=C_RAW/filename
        normalized=(hashlib.sha256(path.read_bytes().replace(b"\r\n",b"\n")).hexdigest()
                    if path.suffix==".json" else None)
        if P.sha(path)!=digest and normalized!=digest:
            raise ValueError("C archive checksum "+filename)
    def ingest(suite,model,gid,size,depth,nodes,observations):
        signature=tuple((n["slot"],n["pos"],n["truth"],tuple(n["parents"])) for n in nodes)
        if graphs.setdefault((suite,size,depth,gid),signature)!=signature:
            raise ValueError(f"Graph mismatch: {suite} {size} {depth} {gid}")
        for key,row in observations:
            ident=(suite,size,depth,gid,*key)
            by_model=raw.setdefault(ident,{})
            if model in by_model:
                raise ValueError("Duplicate scene "+str(ident))
            by_model[model]=row
    for model in ORDER:
        for suite in ("standard","layered","expanded"):
            if suite=="layered" and model not in DISCRETE:
                old=archive(model)
                if old["checkpoint"]!=hashes[model]:
                    raise ValueError("Wrong archive weight")
                cases=defaultdict(list)
                for e in old["records"]:
                    cases[e["case"]].append(e)
                for gid in range(352,368):
                    for depth in (4,8):
                        key=f"{gid}-{depth}"
                        c=dict(old["cases"][key],events=cases[key])
                        if len(c["events"])!=22:
                            raise ValueError(key)
                        ingest(suite,model,gid,len(c["nodes"]),depth,c["nodes"],grid_observations(c,True))
                continue
            c_archive=suite=="standard" and model not in DISCRETE
            folder=C_RAW/"raw"/model if c_archive else root/suite/model
            identity=P.read(folder/"identity.json")
            if identity["checkpoint"]!=hashes[model] or identity["model"]!=model:
                raise ValueError("Wrong weight "+str(folder))
            if suite=="standard":
                cells=[(n,d,g) for n in (32,64,128) for d in (4,10) for g in range(32)]
            else:
                ids=range(352,368) if suite=="layered" else range(400,432)
                cells=[(0,d,g) for g in ids for d in (4,8)]
            for n,d,g in cells:
                data=saved(folder/f"{n}-{d}-{g}.json.gz")
                if data["gid"]!=g or data["depth"]!=d or n and len(data["nodes"])!=n:
                    raise ValueError("Unexpected graph")
                size=len(data["nodes"])
                read=standard_observations if suite=="standard" else grid_observations
                ingest(suite,model,g,size,d,data["nodes"],read(data))
                if suite=="standard":
                    for t,f in data["fit"].items():
                        fit.append(dict(model=model,size=n,depth=d,gid=g,t=t,**f))
                    events=data["natural"]
                    replay=[z for z in events if "replay_child" in z]
                    confirmed=[z for z in replay if z["specific_rescue"]]
                    def spontaneously(z):
                        pair=[data["gold"][0][i] for i in (z["u"],z["q"])]
                        return any(x==y==pair for x,y in zip(z["joint_followup"],z["joint_followup"][1:]))
                    natural.append(dict(model=model,size=n,depth=d,gid=g,opportunities=data["natural_opportunities"],
                        errors=len({(z["t"],z["u"]) for z in events}),replays=len(replay),
                        replay_rescues=len(confirmed),with_two=len([z for z in confirmed if len(z["joint_followup"])>=2]),
                        natural_two=sum(spontaneously(z) for z in confirmed)))
    for (suite,n,d,g,role,phase,condition),models in raw.items():
        if set(models)!=set(ORDER):
            raise ValueError("Missing matched model")
        common=all(z["eligible"] for z in models.values())
        same=common and len({z["t"] for z in models.values()})==1
        for model,z in models.items():
            for scope,ok in (("individual",z["eligible"]),("common",common),("same_round",same)):
                for size,depth in ((n,d),(0,0)):
                    cell=(suite,size,depth,role,phase,condition,scope,model)
                    denom[cell].update(attempts=1,eligible=int(ok))
                    if ok:
                        evidence[cell].update({k:int(v) for k,v in z.items() if k not in ("eligible","t") and v is not None})
    counts={}
    for k,den in denom.items():
        counts[k]=den.copy()
        counts[k].update(evidence[k])
    return counts,fit,natural


def examples(root):
    lines=["## 同一道题怎样变化","","下列例子取 32 节点、深度 4 档中按编号最早的四方共同合格题。括号内依次为目标值和子节点值，空表示尚无数值。每条轨迹从写入后的输入开始，随后列出前三轮与第十六轮。",""]
    def pair(z):
        def cell(v):
            return "空" if v==24 else "不适用" if v in (23,None) else str(v)
        return "（"+cell(z["u"])+"，"+cell(z["q"])+"）"
    for phase,cond in (("first_correct","target"),("graph_complete","target"),("first_correct","parent")):
        for gid in range(32):
            group=[]
            for m in ORDER:
                folder=(root/"standard"/m) if m in DISCRETE else C_RAW/"raw"/m
                case=saved(folder/f"32-4-{gid}.json.gz")
                group.append(next(e for e in case["events"] if
                    (e["role"],e["timing"],e["condition"])==("internal",phase,cond)))
            if any("skip" in e for e in group):
                continue
            lines.extend(["### "+CN[phase]+"，"+CN[cond]+"（图 "+str(gid)+"）","",
                "| 模型 | 原目标与子节点真值 | 写入后 → 第1轮 → 第2轮 → 第3轮 → 第16轮 | 同起点干净分支第1轮 |",
                "|---|---|---|---|"])
            for name,e in zip(NAMES,group):
                path=[e["inputs"][0],*e["trace"][:3],e["trace"][-1]]
                lines.append("| "+name+" | "+str((e["gold_target"],e["gold_child"]))+" | "+
                             " → ".join(pair(z) for z in path)+" | "+pair(e["clean"][0])+" |")
            break
        else:
            lines.append(CN[phase]+"，"+CN[cond]+"：该档没有共同合格题。")
        lines.append("")
    return lines


def report(root,out):
    counts,fit,natural=aggregate(root)
    out.mkdir(parents=True,exist_ok=True)
    cols=("suite","size","depth","role","timing","condition","scope","model")
    csv_output(out/"scenes.csv",[dict(zip(cols,k),**den) for k,den in counts.items()])
    csv_output(out/"same-graph-fit.csv",fit)
    csv_output(out/"natural.csv",natural)
    def ratio(z,num,den):
        return f"{z[num]}/{z[den]}" if z[den] else "—"
    lines=["# 原始 C 与离散 A′、B 在相同注错场景中的行为","",
        "考察普通连续模型 C 在哪些场景可以改写已经算过的结果，以及这项性质在同架构离散 A′ 和带纠错训练的 B 上分别怎样表现。A′、B 为种子 42 的 e86、e92；两个原始 C 为种子 42、1234 的 e143。推断范围限于这些权重与实际合格的题例。","",
        "## 按场景比较","","每格依次给出三个原始计数：目标在下一轮正确／共同合格数，子节点相对干净分支出现额外错误／干净分支子节点曾正确数，发生额外错误后末两轮两处均正确／有足够随访的额外受损数。空位在干净分支应有正确值时也按受损计。终点没有子节点。各模型在每行使用相同题例；绝对轮次也相同的子集、个别权重资格及每档大小和深度另见 scenes.csv。",""]
    brief=["## 已算位置能否重新写对","",
           "每项在相同合格题例上比较。对内部节点只改其已算对的值，保留原图与父值。连续模型换对应位置的隐藏向量，离散模型只换数值字段。",""]
    for phase in ("first_correct","child_correct","graph_complete"):
        group=[counts[("standard",0,0,"internal",phase,"target","common",m)] for m in ORDER]
        if not group[0]["eligible"]:
            brief.append("- "+CN[phase]+"，四方共同合格 0 例。")
            continue
        actual="，".join(name+" "+ratio(z,"next_correct","eligible") for name,z in zip(NAMES,group))
        stable="，".join(name+" "+ratio(z,"specific_end","specific_follow") for name,z in zip(NAMES,group))
        brief.append("- "+CN[phase]+"后改错。下一轮重新算对的次数为"+actual+"。在子节点确实采用特定错值且尚有两轮可观察的题例中，末两轮两处均正确为"+stable+"。")
    brief.extend(["","下表逐场景展开。父值干预时，“目标”仍指它的子节点；分层与扩展表固定跟踪链中内部节点及其直接子节点。注入点本身、输入源和输出终点的逐轮轨迹可回到原始记录核对。",""])
    lines[4:4]=brief+examples(root)
    def section(suite,heading):
        lines.extend(["### "+heading,"","| 注错场景 | "+" | ".join(NAMES)+" |","|---|---|---|---|---|"])
        clock=("early","before","after","complete","first_correct","child_correct","graph_complete")
        keys=sorted({(k[3],k[4],k[5]) for k in counts if k[0]==suite},
                    key=lambda k:(clock.index(k[1]),k[0],k[2]))
        for role,t,cond in keys:
            if suite=="layered":
                name=LEGACY.get(cond,"干净对照")
            elif suite=="expanded":
                name=CN[cond.removeprefix(t+"_")]
            else:
                name=CN[role]+"，"+CN[cond]
            label=CN[t]+"，"+name
            cells=[]
            for model in ORDER:
                z=counts[(suite,0,0,role,t,cond,"common",model)]
                cells.append(" · ".join(ratio(z,a,b) for a,b in (
                    ("next_correct","eligible"),("child_damage","q_risk"),("end_joint","follow"))))
            lines.append("| "+label+" | "+" | ".join(cells)+" |")
        lines.append("")
    section("standard","32、64、128 节点，按实际求解进度注错")
    section("layered","原始 32 题例，逐层重做相同的 22 种操作")
    section("expanded","64 张新题例，输入位置、两父值、两条分支与连续层")
    lines.extend(["## 自然错误的对照","","| 权重 | 正确后变成错数的节点轮次／机会 | 人工回放上一轮状态后救回子节点／回放边 | 其中无人干预曾连续两轮两处均正确／有至少两轮随访 |",
                  "|---|---|---|---|"])
    for model,name in zip(ORDER,NAMES):
        z=Counter()
        for row in natural:
            if row["model"]==model:
                z.update({k:v for k,v in row.items() if k not in ("model","size","depth","gid")})
        lines.append("| "+name+" | "+" | ".join(ratio(z,a,b) for a,b in (
            ("errors","opportunities"),("replay_rescues","replays"),("natural_two","with_two")))+" |")
    lines.extend(["","## 同题基础求解","","在各档标准轮数（最大依赖深度加一）后再运行四轮，统计干净图的节点数值与前四字段同时正确数。两个模型基础拟合有差异的档位应单独解释。","",
                  "| 节点数／深度 | "+" | ".join(NAMES)+" |","|---|---|---|---|---|"])
    for n in (32,64,128):
        for d in (4,10):
            cells=[]
            for model in ORDER:
                rows=[r for r in fit if r["model"]==model and r["size"]==n and r["depth"]==d and int(r["t"])==d+5]
                total=sum(r["nodes"] for r in rows)
                cells.append(f"{sum(r['value'] for r in rows)}/{total} · {sum(r['first4'] for r in rows)}/{total}")
            lines.append(f"| {n}／{d} | "+" | ".join(cells)+" |")
    dest=out/"场景对照实验报告.md"
    dest.write_text("\n".join(lines)+"\n",encoding="utf8")
    return dest


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root",type=Path,default=OUT/"formal")
    ap.add_argument("--out",type=Path,default=OUT/"report")
    args=ap.parse_args()
    print(report(args.root,args.out))
