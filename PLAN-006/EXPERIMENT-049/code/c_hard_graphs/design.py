"""Paired modulo-23 addition programs, entirely specified before model inference."""
from collections import Counter
from pathlib import Path
import random
import sys

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "c_local_update"))
from rebinding import P, render

SCALES = [(4, 8), (8, 8), (4, 15), (8, 15)]
PROFILES = ("single_long", "uneven", "balanced", "all_long")
EXTENSION = (8, 20, "all_long")
PERIODS = (0, 4, 2)  # 0: fresh variable for each definition
ARMS = ("unique", "reuse4", "reuse2")
COUNT = 32
PROBES = 8
EXTRA = 16


def lengths(width, depth, profile):
    uneven = [depth, depth - 2] + [2] * (width - 2)
    if profile == "single_long":
        return [depth] + [2] * (width - 1)
    if profile == "uneven":
        return uneven
    if profile == "all_long":
        return [depth] * width
    if profile != "balanced":
        raise ValueError(profile)
    base, extra = divmod(sum(uneven[1:]), width - 1)
    return [depth] + [base + int(i < extra) for i in range(width - 1)]


def make_case(width, depth, profile, gid):
    lens = lengths(width, depth, profile)
    def rng(*key):
        return random.Random(P.seed("hard-graphs-v1", width, depth, gid, *key))
    constants = rng("constants").sample(range(1, 23), 6)
    attrs = [(-1, 0)] * 6
    parents = [[] for _ in constants]
    latest = [0] * width
    for d in range(1, depth + 1):
        order = list(range(width))
        rng("order", d).shuffle(order)
        for lane in order:
            if d > lens[lane]:
                continue
            prev = latest[lane] if d > 1 else lane % 6
            k = rng("operand", lane, d).choice([j for j in range(1, 6) if j != prev])
            parents.append([prev, k])
            attrs.append((lane, d))
            latest[lane] = len(attrs) - 1
    names = [f"x_{i}" for i in rng("names").sample(range(128), 128)]
    arms = {}
    for arm, period in zip(ARMS, PERIODS):
        if not period and len(attrs) > len(names):
            continue
        slots = [f"k{i}" for i in range(6)]
        ops = list(zip(slots, constants))
        for i, (lane, d) in enumerate(attrs[6:], 6):
            slot = f"s{lane}_{(d-1) % period if period else d-1}"
            ops.append((slot, tuple(slots[j] for j in parents[i])))
            slots.append(slot)
        # Allocate the name of the unique arm's first definition to each slot.
        first = list(dict.fromkeys(slots))
        naming = ([names[slots.index(s)] for s in first] if len(attrs) <= len(names)
                  else names[:len(first)])
        branch = render(ops, naming, False)
        nodes = branch["nodes"]
        if [n["parents"] for n in nodes] != parents or len(branch["tokens"]) > 1024:
            raise ValueError("invalid binding or sequence length")
        arms[arm] = branch
    u = attrs.index((0, depth // 2))
    q = attrs.index((0, depth // 2 + 1))
    return dict(width=width, depth=depth, profile=profile, gid=gid, lengths=lens,
                layers=dict(Counter(d for _, d in attrs)), attrs=attrs, u=u, q=q,
                D=depth+1, arms=arms)
