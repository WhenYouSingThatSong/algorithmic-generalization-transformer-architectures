"""Score actual per-node inputs and layer-two predictions against synchronous local truth."""
from collections import Counter, defaultdict

from local_protocol import classify, P


def metrics(g, y, trace, clean, timing, condition, target):
    nodes = g["nodes"]
    index = {p: i for i, p in enumerate(nodes)}
    truth = [y[p][3] for p in nodes]
    pending = set()
    pending_seen = set()
    bins = defaultdict(Counter)
    phases = []
    first_groups = []
    for t, row in enumerate(trace):
        groups = []
        decisions = []
        events = []
        for i, p in enumerate(nodes):
            old = row["input"][i][3]
            parents = [row["input"][index[k]][3] for k in g["parents"][p]]
            ready = all(0 <= v < 23 for v in parents)
            expected = (sum(parents) % 23 if parents else y[g["constants"][p]][3]) if ready else old
            category = classify(old, expected, ready, truth[i])
            groups.append(category)
            z = row["update"][i]
            change = old != expected
            if z["reference"] != change:
                raise ValueError("local truth disagrees with evaluator")
            out = row["output"][i][0]
            ok = out == expected
            decisions.append(ok and z["plan"] == change)
            event = Counter(observed=1, reference=int(change), planned=int(z["plan"]),
                            tp=int(z["plan"] and change), fp=int(z["plan"] and not change),
                            fn=int(change and not z["plan"]), execute_correct=int(ok),
                            changed=int(z["changed"]), aligned=int(z["plan"]==z["changed"]),
                            jointly_correct=int(decisions[-1]), gold_correct=int(out==truth[i]),
                            damaged_vs_clean=int(clean[t]["output"][i][0]==truth[i] and out!=truth[i]))
            if "cleared" in z:
                event["cleared"] = int(z["cleared"])
            events.append(event)
            for scope in (["all", "post1"] if t==0 else ["all"]):
                bins[(category, scope)].update(event)
                if target is not None and p == target:
                    bins[(category, "target_" + scope)].update(event)
            if category == "indirect_consistent" and i not in pending_seen:
                pending_seen.add(i)
                pending.add(i)
            elif i in pending and category == "direct_conflict":
                bins[("indirect_first_conflict", "all")].update(event)
                pending.remove(i)
            elif i in pending and out == truth[i]:
                pending.remove(i)
        counts = Counter(groups)
        mixed = all(counts[k] for k in ("new_calculation", "direct_conflict", "consistent_correct"))
        if t == 0:
            for category,event in zip(groups,events):
                bins[(category,"mixed_post1" if mixed else "not_mixed_post1")].update(event)
        phases.append(dict(round=t+1, mixed=mixed, roles=dict(counts),
                           jointly_correct=all(ok for group,ok in zip(groups,decisions) if group in
                                               ("new_calculation","direct_conflict","consistent_correct"))))
        if t == 0:
            first_groups = groups
    bins[("indirect_wait", "all")]["ever_indirect"] = len(pending_seen)
    role_sets = {"all": nodes}
    if target is not None:
        role_sets.update(target=[target], descendants=P.descendants(g, target),
                         children=[p for p in nodes if target in g["parents"][p]])
    endpoints = {}
    for role, positions in role_sets.items():
        totals = Counter()
        for p in positions:
            i = index[p]
            flags = [z["output"][i][0] == truth[i] for z in trace]
            harms = [j for j, ok in enumerate(flags) if not ok and clean[j]["output"][i][0]==truth[i]]
            totals.update(observed=1, wrong_rounds=len(harms), damaged=bool(harms),
                          last_two=all(flags[-2:]))
            injected_bad = (trace[0]["input"][i][3]!=truth[i] and clean[0]["input"][i][3]==truth[i])
            if injected_bad:
                totals.update(input_damaged=1, first_round_repaired=flags[0])
                if not harms:
                    totals["damaged"] += 1
            if harms or injected_bad:
                onset = -1 if injected_bad else harms[0]
                later = flags[onset+1:]
                restored = next((j for j, ok in enumerate(later, 1) if ok), None)
                totals.update(restored=restored is not None,
                              restoration_delay=restored or 0,
                              stable_after_damage=len(later)>=2 and all(later[-2:]),
                              reerror=restored is not None and not all(later[restored-1:]))
        endpoints[role] = dict(totals)
    return dict(timing=timing, condition=condition, target_class=(
        first_groups[index[target]] if target is not None else None), rounds=phases,
        metrics=[dict(category=k, scope=s, counts=dict(v)) for (k,s),v in bins.items()],
        endpoints=endpoints)


def summarize(records):
    pooled = defaultdict(Counter)
    macros = defaultdict(list)
    coverage = Counter()
    mixed = defaultdict(Counter)
    ratios = dict(precision=("tp","planned"), recall=("tp","reference"),
                  local_accuracy=("execute_correct","observed"), joint_accuracy=("jointly_correct","observed"),
                  false_alarm_rate=("fp","observed"), damage_rate=("damaged_vs_clean","observed"))
    for r in records:
        prefix = (r["split"], r["timing"], r["condition"])
        coverage[prefix + (r["status"],)] += 1
        if r["status"] != "ok":
            continue
        m = mixed[prefix]
        m["graphs"] += 1
        if r["rounds"][0]["mixed"]:
            m["mixed_at_input"] += 1
            m["mixed_all_decisions_correct"] += r["rounds"][0]["jointly_correct"]
        for item in r["metrics"]:
            key = prefix + (item["category"],item["scope"])
            c = item["counts"]
            pooled[key].update(c)
            for name, (a,b) in ratios.items():
                denominator = c.get(b,0)
                if name == "false_alarm_rate":
                    denominator -= c.get("reference",0)
                if denominator:
                    macros[key+(name,)].append(c.get(a,0)/denominator)
        for role, c in r["endpoints"].items():
            key = prefix + ("trajectory", role)
            pooled[key].update(c)
            for name,a,b in (("damage_rate","damaged","observed"), ("stable_given_damaged","stable_after_damage","damaged"),
                             ("delay_given_restored","restoration_delay","restored"), ("reerror_given_restored","reerror","restored")):
                if c.get(b,0):
                    macros[key+(name,)].append(c.get(a,0)/c[b])
    return dict(coverage=[dict(key=k, graphs=v) for k,v in coverage.items()],
                mixed=[dict(key=k, counts=dict(v)) for k,v in mixed.items()],
                pooled=[dict(key=k, counts=dict(v)) for k,v in pooled.items()],
                graph_equal=[dict(key=k, graphs=len(v), mean=sum(v)/len(v)) for k,v in macros.items()])
