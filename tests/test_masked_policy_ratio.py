import unittest

import torch

from src.rl.grpo import GRPOEngine


class MaskedPolicyRatioTest(unittest.TestCase):
    def test_excluded_log_ratio_cannot_overflow_loss_or_gradient(self):
        engine = GRPOEngine(None, None, None, clip_param=0.2)
        for excluded_old in (-1.0, -1001.0):
            with self.subTest(excluded_old=excluded_old):
                current = torch.tensor([[-1.0, -1.0]], requires_grad=True)
                old = torch.tensor([[-1.0, excluded_old]])
                loss = engine.ppo_loss(
                    current, old, -torch.ones_like(current),
                    torch.zeros_like(current), torch.tensor([[1.0, 0.0]]),
                )
                (gradient,) = torch.autograd.grad(loss, current)
                torch.testing.assert_close(loss, torch.tensor(1.0))
                torch.testing.assert_close(gradient, torch.tensor([[1.0, 0.0]]))
