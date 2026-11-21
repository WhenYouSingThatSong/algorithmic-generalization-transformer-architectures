"""Re-score saved dense trajectories and collect prospective, fixed-bin signals."""
from collections import Counter, defaultdict

import torch

from design import EXTRA


def category(value, truth):
    if value == truth:
        return "correct"
    if 0 <= value < 23:
        return "wrong_number"
    return "empty" if value == 24 else "non_numeric"


def transitions(case, traces):
    """Each transition belongs to one bin. First observed output has no predecessor."""
    counts = defaultdict(Counter)
    for arm, trace in traces.items():
        nodes = case["arms"][arm]["nodes"]
        if len(trace) != case["D"] + EXTRA or any(len(obs) != len(nodes) for obs in trace):
            raise ValueError("incomplete dense trace")
        ever_correct = [z[0] == node["truth"] for z, node in zip(trace[0], nodes)]
        for z, node in zip(trace[0], nodes):
            first = category(z[0], node["truth"])
            counts[arm,"first_output","unobserved",first].update(
                observations=1, first_correct=int(first=="correct"), returned_to_correct=0,
                variable_correct_to_wrong=0, numeric_correct_at_both=0,
                identity_lost_while_value_correct=0)
        for completed, (before, after) in enumerate(zip(trace, trace[1:]), 1):
            window = "within_standard" if completed < case["D"] else "additional_rounds"
            for i, (a, b, node) in enumerate(zip(before, after, nodes)):
                truth = node["truth"]
                old, new = category(a[0], truth), category(b[0], truth)
                c = counts[arm, window, old, new]
                c["observations"] += 1
                c["first_correct"] += new == "correct" and not ever_correct[i]
                c["returned_to_correct"] += new == "correct" and old != "correct" and ever_correct[i]
                c["variable_correct_to_wrong"] += bool(a[2]) and not b[2]
                c["numeric_correct_at_both"] += old == new == "correct"
                c["identity_lost_while_value_correct"] += old == new == "correct" and bool(a[2]) and not b[2]
                ever_correct[i] |= new == "correct"
    return [dict(arm=k[0], window=k[1], previous=k[2], following=k[3], **c)
            for k, c in sorted(counts.items())]


def signal_rows(rt, case):
    """Output at t predicts the t+1 transition; UPDATE is sampled inside t+1."""
    names = list(case["arms"])
    branches = list(case["arms"].values())
    pos = [n["pos"] for n in branches[0]["nodes"]]
    truth = torch.tensor([n["truth"] for n in branches[0]["nodes"]], device=rt.device)
    x = torch.tensor([rt.tok.encode_tokens(b["tokens"]) for b in branches], device=rt.device)
    h = rt.init(x)
    prev = previous_value = margin = None
    bins = defaultdict(Counter)
    for t in range(1, case["D"] + EXTRA + 1):
        h, logits, middle = rt.forward(h, middle=True)
        value_logits = logits[3][:, pos].float()
        top = value_logits.topk(2, dim=-1)
        now = top.indices[..., 0]
        if prev is not None:
            # Margin in logits, including EMPTY/N/A competitors. No gold used to bin.
            bucket = (margin >= 1).long() + (margin >= 4).long()
            stable = (torch.full_like(prev, 2) if previous_value is None else (prev == previous_value).long())
            plan = middle[:, pos].argmax(-1)
            old_status = torch.where(prev == truth, 0, torch.where(prev < 23, 1, 2))
            cell = (((bucket * 3 + stable) * 2 + plan) * 3 + old_status) * 2 + (now == truth).long()
            phase = "within_standard" if t <= case["D"] else "additional_rounds"
            for name, sample in zip(names, cell.tolist()):
                bins[name, phase].update(sample)
        previous_value, prev = prev, now
        margin = top.values[..., 0] - top.values[..., 1]
    rows = []
    for (name, phase), counts in sorted(bins.items()):
        for value, n in sorted(counts.items()):
            value, next_ok = divmod(value, 2)
            value, old = divmod(value, 3)
            value, update = divmod(value, 2)
            margin_bin, stable = divmod(value, 3)
            rows.append(dict(arm=name, window=phase, margin_bin=("below_1","1_to_4","4_plus")[margin_bin],
                             repeat=("changed","same","no_history")[stable], update=update,
                             previous=("correct","wrong_number","non_numeric")[old],
                             next_correct=next_ok, observations=n))
    return rows
