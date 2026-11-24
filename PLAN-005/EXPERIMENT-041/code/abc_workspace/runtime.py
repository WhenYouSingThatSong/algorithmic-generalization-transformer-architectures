"""Minimal shared model adapter. C carries hidden vectors between forwards."""
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


class Runtime:
    def __init__(self, spec, device):
        sys.path.insert(0, str(HERE.parent/'b_mechanism'))
        from trajectory_io import configure_runtime
        configure_runtime()
        import torch
        import b_mechanism_common as B
        self.torch = torch
        mc, tc, dc = B.load_configs(spec['config'])
        self.tok = B.build_factored_tokenizer(dc)
        if self.tok.mod_val != 23 or self.tok.n_vars < 128: raise ValueError('mod23 / 128 names required')
        mc.vocab_sizes, mc.factors = self.tok.vocab_sizes, self.tok.factors
        teacher = B.DAGTeacherModel(mc, self.tok)
        self.lit = B.LitRecurrentTransformerModel(mc, dc, tc, teacher)
        self.lit.load_ckpt(spec['path'])
        self.device = torch.device(device)
        self.lit.to(self.device).eval()
        self.continuous = bool(self.lit.nointerm)
        if self.continuous != (spec['family'] == 'C'): raise ValueError('family / feedback mismatch')
        self.forwards = 0
        torch.set_float32_matmul_precision('medium')

    def tokens(self, layout, length=256):
        rows = self.tok.encode_tokens(layout['tokens'])
        if len(rows) > length: raise ValueError('no truncation allowed')
        rows += [self.tok.encode_token('<PAD>')] * (length - len(rows))
        return self.torch.tensor([rows], device=self.device, dtype=self.torch.long)

    def init(self, raw):
        return self.lit.model.token_embedder(raw) if self.continuous else raw.clone()

    def decode(self, state):
        if not self.continuous: return state
        return self.torch.stack([h(state).argmax(-1) for h in self.lit.model.output_layers], -1)

    def write_rows(self, state, raw, rows):
        out = state.clone()
        if rows: out[:, rows] = self.init(raw)[:, rows]
        return out

    def reset_results(self, state, raw, rows):
        # C: whole hidden row reinitialization, NOT an exact VALUE-only reset.
        if self.continuous: return self.write_rows(state, raw, rows)
        out = state.clone()
        out[:, rows, 3] = self.tok.value_tok2idx['EMPTY']
        return out

    def run(self, start, rounds, length):
        state = start.clone(); inputs = []; outputs = []
        for _ in range(rounds):
            inputs.append(self.decode(state)[0, :length, :4].cpu().tolist())
            if self.continuous:
                state, logits = self.lit.model(state, True)
                pred = self.torch.stack([x.argmax(-1) for x in logits], -1)
            else:
                logits = self.lit.model(state)
                state = self.torch.stack([x.argmax(-1) for x in logits], -1)
                state[..., 4:] = 0
                pred = state
            self.forwards += 1
            outputs.append(pred[0, :length, :4].cpu().tolist())
        return state, dict(inputs=inputs, outputs=outputs)

    def truth(self, raw, positions, values):
        rows = raw[0, :, :4].cpu().tolist()
        for p, v in zip(positions, values): rows[p][3] = self.tok.value_tok2idx[str(v)]
        return rows

    def numeric_commit(self, state, positions):
        row = self.decode(state)[0].cpu().tolist()
        vals = [self.tok.value_idx2tok[r[3]] for r in (row[p] for p in positions)]
        return [int(v) if v.isdigit() else None for v in vals]
