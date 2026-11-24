"""Small tensor checks, run by server agent with torch; skipped on review host."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'torch needed on execution server')
class TrainingTests(unittest.TestCase):
    def test_continuation_gradient_and_final_labels(self):
        from train import recurrent_loss
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.w = torch.nn.Parameter(torch.tensor(0.8))
                self.embed = torch.nn.Parameter(torch.tensor(0.3))
                self.embed_calls = 0
                self.states = []
            def token_embedder(self, x):
                self.embed_calls += 1
                return self.embed.expand(x.shape[0], x.shape[1], 1)
            def forward(self, h, nointerm):
                self.assert_continuous = nointerm
                h = h * self.w + 0.1
                h.retain_grad(); self.states.append(h)
                return h, [torch.cat([h, -h], -1)]
        model = Model()
        seen = []
        def criterion(logits, labels):
            seen.append(labels.tolist())
            loss = torch.nn.functional.cross_entropy(logits, labels)
            # Only the continuation contributes gradients in this fixture.
            return loss * 0 if len(seen) <= 2 else loss
        def teachers(x, y, depths):
            # D=2: stage1 class0, stage2 class1, no extra teacher entries.
            return torch.tensor([[[[0]]], [[[0]]], [[[1]]]]), None, None
        lit = SimpleNamespace(model=model, get_teacher_batch_with_depth=teachers,
                              criterion=criterion, factor_loss_scales=[1])
        inputs = torch.zeros(1,1,1,dtype=torch.long)
        loss, _, _, d = recurrent_loss(lit, inputs, inputs, torch.tensor([[[1]]]), 2)
        self.assertEqual(d,2); self.assertEqual(model.embed_calls,1)
        self.assertEqual(seen,[[0],[1],[1],[1],[1],[1]])
        loss.backward()
        self.assertIsNotNone(model.embed.grad)
        self.assertGreater(float(model.embed.grad.abs()),0)
        self.assertGreater(float(model.states[0].grad.abs().sum()),0)
        self.assertTrue(all(s.grad is not None for s in model.states))
        self.assertTrue(model.assert_continuous)


if __name__ == '__main__':
    unittest.main()
