"""Pure planning for W1/W2. Old/new truth never writes computed answers into inputs."""
import copy
import hashlib

BRANCHES=("clean","sham","state_error","new_warm","new_scratch","new_warm_pinned","new_scratch_pinned")
DELTAS=(1,3,7,11)

def truth(nodes, depths, parents, constants, modulus=23):
    values=dict(constants)
    for p in sorted(nodes,key=lambda p:(depths[p],p)):
        ps=parents[p]
        if not ps or any(x not in values for x in ps):
            raise ValueError("missing/out-of-order DAG inputs")
        values[p]=sum(values[x] for x in ps)%modulus
    return values

def descendants(nodes, parents, leaf):
    seen={leaf}
    for _ in nodes:
        added={q for q in nodes if any(p in seen for p in parents[q])}
        if added <= seen:break
        seen |= added
    return sorted(seen-{leaf})

def select_leaf(nodes, depths, parents, constants, gid, seed=20261118):
    candidates=[(p,parents[p][0]) for p in nodes if depths[p]==0 and len(parents[p])==1
                and parents[p][0] in constants and descendants(nodes,parents,p)]
    return min(candidates,key=lambda pc:hashlib.sha256(f"{seed}:{gid}:{pc[0]}:{pc[1]}".encode()).hexdigest()) if candidates else None

def make_plan(nodes, depths, parents, constants, old, leaf, const, delta, modulus=23):
    if delta%modulus==0:raise ValueError("zero effective delta")
    recomputed=truth(nodes,depths,parents,constants,modulus)
    if any(recomputed[p]!=old[p] for p in nodes):raise ValueError("old labels fail independent ADD evaluator")
    new_constants=dict(constants);new_constants[const]=(constants[const]+delta)%modulus
    new=truth(nodes,depths,parents,new_constants,modulus)
    ds=descendants(nodes,parents,leaf)
    affected=[p for p in nodes if new[p]!=old[p]]
    return dict(leaf=leaf,constant=const,delta=delta,new_constant=new_constants[const],
        old_truth={p:recomputed[p] for p in nodes+[const]},new_truth={p:new[p] for p in nodes+[const]},
        regions=dict(all_nodes=list(nodes),leaf=[leaf],changed_nodes=affected,
                     unchanged_nodes=[p for p in nodes if p not in affected],
                     changed_descendants=[p for p in ds if p in affected],
                     unchanged_descendants=[p for p in ds if p not in affected],
                     outside_descendants=[p for p in nodes if p!=leaf and p not in ds],constant=[const]))

def starts(base, raw, plan, value_index, encode):
    """base/raw are integer token rows. Only one VALUE cell is changed in warm arms."""
    result={}
    for name in BRANCHES:
        scratch="scratch" in name
        x=copy.deepcopy(raw if scratch else base)
        if name=="sham":
            x[plan["leaf"]][value_index]=base[plan["leaf"]][value_index]
        elif name=="state_error":
            x[plan["leaf"]][value_index]=encode(plan["new_truth"][plan["leaf"]])
        elif name.startswith("new_"):
            x[plan["constant"]][value_index]=encode(plan["new_constant"])
        result[name]=dict(state=x,truth_ref="new" if name.startswith("new_") else "old",
            start_kind="scratch" if scratch else "completed",
            pinned=name.endswith("_pinned"))
    return result
