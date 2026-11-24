"""Structural multi-parent sampling, independent of model success."""
import hashlib


def select_targets(parents, depths, graph_id, seed=20261115):
    selected, coverage = [], []
    has_child = {p for ps in parents.values() for p in ps}
    for fan_in in (1, 2, 3):
        pool = [p for p,ps in parents.items() if len(ps)==fan_in
                and depths[p]>=1 and p in has_child]
        rank = lambda p: hashlib.sha256(f'{seed}|{graph_id}|{fan_in}|{p}'.encode()).hexdigest()
        chosen = min(pool,key=rank) if pool else None
        coverage.append(dict(fan_in=fan_in,candidates=sorted(pool),selected=chosen))
        if chosen is not None:
            selected.append(chosen)
    return selected, coverage


def value_branches(truth, modulus, empty_id, exhaustive=False):
    deltas = range(1,modulus) if exhaustive else (1,3,7,11)
    return [('unchanged',None),('correct_write',truth),('empty',empty_id)]+[
        (f'wrong_plus_{d}',(truth+d)%modulus) for d in deltas]
