"""Finite excluded tokens must not affect GRPO loss or gradient.

Tiny CPU fixtures use the actual repository's GRPOEngine.ppo_loss and, after
the fix, the public helper used by train_mvp.py. No model weights/downloads.
"""
import unittest

import torch

from src.rl import grpo


class MaskedGRPONumericsTests(unittest.TestCase):
    def setUp(self):
        self.engine = grpo.GRPOEngine(None, None, None, beta=0.04, clip_param=0.2)

    def _loss_with_policy_gap(self, dtype, *, masked_gap, advantage=-1.0):
        policy = torch.nn.Parameter(torch.tensor([[-1.0, -1.0]], dtype=dtype))
        old = torch.tensor([[-1.0, -1.0 - masked_gap]], dtype=dtype)
        advantages = torch.tensor([[advantage, advantage]], dtype=dtype)
        kl = torch.zeros((1, 2), dtype=dtype)
        mask = torch.tensor([[1.0, 0.0]], dtype=dtype)
        loss = self.engine.ppo_loss(policy, old, advantages, kl, mask)
        loss.backward()
        return loss, policy.grad

    def test_normal_active_token_matches_existing_loss_and_gradient(self):
        loss, grad = self._loss_with_policy_gap(torch.float64, masked_gap=0)
        torch.testing.assert_close(loss, torch.tensor(1.0, dtype=torch.float64))
        torch.testing.assert_close(grad, torch.tensor([[1.0, 0.0]], dtype=torch.float64))

    def test_masked_policy_ratio_overflow_ignored_in_forward_and_backward(self):
        for dtype, gap in (
            (torch.float16, 20.0),
            (torch.bfloat16, 120.0),
            (torch.float32, 120.0),
            (torch.float64, 800.0),
        ):
            with self.subTest(dtype=str(dtype)):
                loss, grad = self._loss_with_policy_gap(dtype, masked_gap=gap)
                self.assertTrue(torch.isfinite(loss).item())
                self.assertTrue(torch.isfinite(grad).all().item())
                torch.testing.assert_close(loss.float(), torch.tensor(1.0))
                torch.testing.assert_close(grad.float(), torch.tensor([[1.0, 0.0]]))

    def test_masked_kl_ratio_from_training_producer_is_sanitized_before_exp(self):
        # train_mvp.py must use this helper on current/reference log probabilities.
        policy = torch.nn.Parameter(torch.tensor([[-1.0, -1.0]]))
        reference = torch.tensor([[-1.0, -120.0]])
        mask = torch.tensor([[1.0, 0.0]])
        kl = grpo.masked_kl_penalty(policy, reference, mask, 0.04)
        self.assertTrue(torch.isfinite(kl).all().item())
        torch.testing.assert_close(kl, torch.zeros_like(kl))
        loss = self.engine.ppo_loss(
            policy, policy.detach(), -torch.ones_like(policy), kl, mask
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertTrue(torch.isfinite(policy.grad).all().item())
        torch.testing.assert_close(loss, torch.tensor(1.0))
        torch.testing.assert_close(policy.grad, torch.tensor([[1.0, 0.0]]))

    def test_kl_ordinary_active_input_matches_prior_scalar_formula(self):
        policy = torch.nn.Parameter(torch.tensor([[-1.0, -1.0]], dtype=torch.float64))
        reference = torch.tensor([[-0.7, -0.5]], dtype=torch.float64)
        mask = torch.tensor([[1.0, 0.0]], dtype=torch.float64)
        kl = grpo.masked_kl_penalty(policy, reference, mask, 0.04)
        import math
        expected = 0.04 * (math.exp(-0.3) - (-0.3) - 1.0)
        self.assertAlmostEqual(kl[0, 0].item(), expected, places=12)
        self.assertEqual(kl[0, 1].item(), 0.0)
        kl.sum().backward()
        self.assertEqual(policy.grad[0, 1].item(), 0.0)
        self.assertAlmostEqual(policy.grad[0, 0].item(), 0.04 * (math.exp(-0.3) - 1), places=12)

    def test_masking_does_not_mutate_detached_old_policy_alias(self):
        policy = torch.nn.Parameter(torch.tensor([[-1.0, -1.0]]))
        old = policy.detach()
        original = policy.detach().clone()
        mask = torch.tensor([[1.0, 0.0]])
        kl = torch.zeros_like(policy)
        loss = self.engine.ppo_loss(policy, old, -torch.ones_like(policy), kl, mask)
        loss.backward()
        torch.testing.assert_close(policy.detach(), original)
        torch.testing.assert_close(old, original)
        self.assertEqual(policy.grad[0, 1].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
