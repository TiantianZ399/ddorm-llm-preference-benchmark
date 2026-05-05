from __future__ import annotations

import torch

from benchmark.gpu.ddorm_math import ddorm_target_distribution


def test_reward_centering_invariance() -> None:
    torch.manual_seed(0)
    policy_scores = torch.randn(8, 4)
    rewards = torch.randn(8, 4)
    _, q_centered, _, _ = ddorm_target_distribution(policy_scores, rewards, eta=0.7, temperature=1.3)
    _, q_uncentered, _, _ = ddorm_target_distribution(
        policy_scores, rewards, eta=0.7, temperature=1.3, center_rewards=False
    )
    assert torch.allclose(q_centered, q_uncentered, atol=1e-6)


def test_kl_monotone_in_beta() -> None:
    torch.manual_seed(1)
    policy_scores = torch.randn(32, 5)
    rewards = torch.randn(32, 5)
    betas = [0.0, 0.1, 0.3, 0.7, 1.0]
    kls = []
    for beta in betas:
        _, _, _, stats = ddorm_target_distribution(policy_scores, rewards, eta=beta, temperature=1.0)
        kls.append(float(stats.mean_kl_q_p))
    assert all(a <= b + 1e-7 for a, b in zip(kls, kls[1:]))
