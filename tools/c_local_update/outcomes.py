"""Natural drift on clean rollouts, and paired functional use after readout recovery."""
import json
from collections import defaultdict, Counter


def additional(path):
    natural = defaultdict(Counter)
    uses = defaultdict(dict)
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            r = json.loads(line)
            if r["status"] != "ok":
                continue
            if r["module"] == "clean":
                for i, truth in enumerate(r["truth"]):
                    for name, column in (("value",0),("variable",3),("first4",1)):
                        flags = [z["output"][i][column] for z in r["rows"]]
                        if name=="value":
                            flags = [v==truth[3] for v in flags]
                        for window, segment in (("normal",flags[:r["D"]+1]),
                                                ("after_normal",flags[r["D"]:])):
                            c = natural[name,window]
                            c["nodes"] += 1
                            if not any(segment):
                                continue
                            c["ever_correct"] += 1
                            first = segment.index(True)
                            if all(segment[first:]):
                                continue
                            c["drifted"] += 1
                            after = segment[segment.index(False,first)+1:]
                            c["ever_recovered"] += any(after)
                            c["last_two_after_drift"] += len(after)>=2 and all(after[-2:])
            elif r["module"]=="reuse":
                u = r["nodes"].index(r["u"])
                q = r["nodes"].index(r["new_node"])
                used = all(row["output"][q][0]==r["truth"][q][3] for row in r["rows"][-2:])
                uses[r["gid"],r["lag"]][r["condition"]] = (
                    r["u_readout"][3]==r["truth"][u][3], r["u_first4_correct"], used)
    joint = defaultdict(Counter)
    for (_,lag), arms in uses.items():
        if set(arms)!={"clean","perturbed"}:
            continue
        c = joint[lag]
        c["paired_graphs"] += 1
        if not arms["clean"][2]:
            continue
        c["clean_consumer_capable"] += 1
        v,four,used = arms["perturbed"]
        c[f"value_{int(v)}_consumer_{int(used)}"] += 1
        c[f"first4_{int(four)}_consumer_{int(used)}"] += 1
    return dict(natural=[dict(field=k[0],window=k[1],counts=dict(v)) for k,v in natural.items()],
                use_after_injection=[dict(lag=k,counts=dict(v)) for k,v in sorted(joint.items())])
