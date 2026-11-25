"""Ordinary modulo-addition graphs and paired donors."""
import random

from design import P, render


def family(gid, size, depth):
    """Interleave one traceable spine with independent background expressions."""
    rng = random.Random(P.seed("ordinary-c-recovery-v1", gid, size, depth))
    roots = [(f"k{i}", v) for i,v in enumerate(rng.sample(range(23), 6))]
    ops = roots.copy()
    pool = {f"k{i}": 0 for i in range(1, 6)}
    spine = [0]
    slots = set(rng.sample(range(6, size), depth))
    for i in range(6, size):
        dest = f"v{i}"
        if i in slots:
            parents = (ops[spine[-1]][0], f"k{rng.randrange(2,6)}")
            spine.append(i)
        else:
            available = [s for s,d in pool.items() if d < depth]
            parents = tuple(rng.sample(available, rng.randint(1,3)))
            pool[dest] = 1 + max(pool[p] for p in parents)
        ops.append((dest, parents))
    names = [f"x_{i}" for i in rng.sample(range(128), size)]
    base = render(ops, names, True)
    a,b = rng.sample(range(1,23), 2)
    variants = [base]
    # Same-position, same-time donors. k1 never feeds the observed spine.
    for root,delta in ((1,a),(0,a),(0,b)):
        edited = ops.copy()
        slot, val = ops[root]
        edited[root] = slot, (val+delta)%23
        variants.append(render(edited, names, True))
    return variants, spine
