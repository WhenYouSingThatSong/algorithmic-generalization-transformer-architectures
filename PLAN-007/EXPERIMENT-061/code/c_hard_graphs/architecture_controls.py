"""Matched stage-supervised architecture prototypes on the same modulo-23 graph task."""
import argparse
import math
import sys
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from attribution_cases import exposure_case
from design import P, TOOLS
from research_probes import source_labels

sys.path.insert(0, str(TOOLS.parent / "experiments"))
from tokenizers import FactoredVocabTokenizer

VOCABS = (7, 129, 2, 25)
ARMS = ("loop", "fixed_parameter", "fixed_compute", "written")


def parameter_count(width, stacks):
    # Embeddings, four readouts and a pre-norm block with 4x feed-forward width.
    return 2 * sum(VOCABS) * width + sum(VOCABS) + stacks * (12 * width**2 + 13 * width)


class Control(nn.Module):
    def __init__(self, arm, width, heads, stages):
        super().__init__()
        if arm not in ARMS or width % heads or stages < 2:
            raise ValueError("invalid model dimensions or arm")
        shared = arm in ("loop", "written")
        blocks = 1 if shared else stages
        if arm == "fixed_parameter":
            target = parameter_count(width, 1)
            width = min(range(heads, width + 1, heads),
                        key=lambda d: abs(parameter_count(d, blocks) - target))
        self.width, self.stages, self.arm = width, stages, arm
        self.embeddings = nn.ModuleList(nn.Embedding(v, width) for v in VOCABS)
        self.heads = nn.ModuleList(nn.Linear(width, v) for v in VOCABS)
        self.blocks = nn.ModuleList(nn.TransformerEncoderLayer(
            width, heads, width*4, dropout=0, batch_first=True, norm_first=True
        ) for _ in range(blocks))
        if sum(p.numel() for p in self.parameters()) != parameter_count(width, blocks):
            raise ValueError("parameter matching formula differs from actual model")

    def embed(self, fields):
        h = sum(embedding(fields[..., j]) for j, embedding in enumerate(self.embeddings))
        # Fixed sinusoidal positions, including re-encoded scratch entries.
        phase = torch.arange(h.size(1), device=h.device)[:, None] * torch.exp(
            -math.log(10000) * torch.arange(0, self.width, 2, device=h.device) / self.width)
        pos = torch.empty_like(h[0])
        pos[:, 0::2] = phase.sin()
        pos[:, 1::2] = phase[:, :self.width//2].cos()
        return h + pos

    def forward(self, x, positions, edit=None):
        fields = x[..., :4]
        h = self.embed(fields)
        outputs, lengths = [], []
        for j in range(self.stages):
            lengths.append(h.size(1))
            block = self.blocks[0] if len(self.blocks)==1 else self.blocks[j]
            h = block(h)
            # Written arm emits a new table with one entry per definition each stage.
            loc = positions if self.arm != "written" or j == 0 else torch.arange(
                h.size(1)-len(positions), h.size(1), device=h.device)
            z = h[:, loc]
            if edit and edit["stage"] == j and self.arm != "written":
                h = h.clone()
                h[0, loc[edit["node"]]] = z[1, edit["node"]]
                z = h[:, loc]
            logits = [head(z) for head in self.heads]
            outputs.append(logits)
            if self.arm == "written" and j+1 < self.stages:
                new = torch.stack([a.argmax(-1) for a in logits], -1)
                if edit and edit["stage"] == j:
                    u = edit["node"]
                    new[0, u] = new[1, u]
                fields = torch.cat((fields, new), dim=1)
                h = self.embed(fields)
        return outputs, lengths


def example(gid, device):
    c = exposure_case(gid, 4)
    tok = FactoredVocabTokenizer(128, ["ADD"], 23, 3)
    branches = [c["base"], c["donor"]]
    x = torch.tensor([tok.encode_tokens(b["tokens"]) for b in branches], device=device)
    nodes = c["base"]["nodes"]
    pos = [n["pos"] for n in nodes]
    labels = torch.tensor([[source_labels(x[i], b["nodes"], k) for k in range(len(nodes))]
                           for i,b in enumerate(branches)], device=device)
    depths = torch.tensor([n["depth"] for n in nodes], device=device)
    return c, x, pos, labels, depths


def stage_label(labels, depths, stage, stages):
    budget = math.ceil((stage+1) * (1+int(depths.max())) / stages)
    y = labels.clone()
    y[..., 3].masked_fill_(depths >= budget, 24)
    return y


def evaluation(model, device, gids):
    totals = dict(nodes=0, values=0, first_four=0, graphs=0, graph_values=0,
                  attempted=0, eligible=0, downstream_specific=0)
    for gid in gids:
        c,x,pos,y,_ = example(gid, device)
        pred,lengths = model(x,pos)
        final = torch.stack([a[0].argmax(-1) for a in pred[-1]], -1)
        good = final == y[0]
        totals["nodes"] += len(pos)
        totals["values"] += int(good[:,3].sum())
        totals["first_four"] += int(good.all(-1).sum())
        totals["graphs"] += 1
        totals["graph_values"] += int(good[:,3].all())
        stage = model.stages-2
        u,q = c["u"],c["q"]
        uread = torch.stack([a[:,u].argmax(-1) for a in pred[stage]], -1)
        totals["attempted"] += 1
        if torch.equal(uread, y[:,u]) and good[q,3]:
            totals["eligible"] += 1
            changed,_ = model(x,pos,dict(stage=stage,node=u))
            totals["downstream_specific"] += int(changed[-1][3][0,q].argmax()==c["alt"][q])
    # Multiply-adds per input, counting Q/K/V, out, FFN and both attention products.
    w = model.width
    totals["block_mac_per_input"] = sum(12*n*w*w+2*n*n*w for n in lengths)
    totals["sequence_lengths"] = lengths
    return totals


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--device", default="cpu")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--width", type=int, default=32)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--eval-graphs", type=int, default=4)
    args = p.parse_args()
    if args.steps<1 or args.eval_graphs<1:
        p.error("positive training and evaluation counts required")
    torch.set_num_threads(2)
    args.out.mkdir(parents=True, exist_ok=False)
    settings = dict(vars(args),out=str(args.out),protocol="prototype-v1")
    P.write(args.out/"settings.json",settings)
    for arm in ARMS:
        torch.manual_seed(args.seed)
        model = Control(arm,args.width,args.heads,args.stages).to(args.device)
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
        losses = []
        for step in range(args.steps):
            _,x,pos,y,depths = example(2*step,args.device)
            pred,_ = model(x,pos)
            loss = sum(F.cross_entropy(a.flatten(0,1),stage_label(y,depths,i,args.stages)[...,j].reshape(-1))
                       for i,heads in enumerate(pred) for j,a in enumerate(heads)) / args.stages
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            metrics = evaluation(model,args.device,range(1,2*args.eval_graphs,2))
        torch.save(model.state_dict(),args.out/(arm+".pt"))
        P.write(args.out/(arm+".json"),dict(arm=arm,width=model.width,stages=model.stages,
                parameters=sum(p.numel() for p in model.parameters()),losses=losses,metrics=metrics))


if __name__ == "__main__":
    main()
