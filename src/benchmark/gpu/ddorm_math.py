from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class DDORMStats:
    mean_kl_q_p: torch.Tensor
    mean_target_entropy: torch.Tensor
    mean_policy_entropy: torch.Tensor
    mean_reward_gain_under_rm: torch.Tensor
    mean_abs_centered_reward: torch.Tensor


def ddorm_target_distribution(
    policy_scores: torch.Tensor,
    reward_scores: torch.Tensor,
    *,
    eta: float = 1.0,
    temperature: float = 1.0,
    center_rewards: bool = True,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, DDORMStats]:
    """Build the DDO-RM finite-candidate target distribution.

    Args:
        policy_scores: Tensor of shape [batch, K]. Usually average response logprobs.
        reward_scores: Tensor of shape [batch, K]. Scalar reward model scores.
        eta: Decision step size in score space.
        temperature: Softmax temperature for the candidate distribution.
        center_rewards: If True, use r_i - E_p[r]. This leaves exact q unchanged
            up to numerical precision but improves interpretation and conditioning.
        eps: Numerical epsilon for logs.

    Returns:
        p: Current candidate distribution.
        q: DDO-RM target distribution, detached by the caller if used as a target.
        centered_reward: Reward vector after subtracting the p-weighted mean.
        stats: Diagnostic tensors for logging.
    """
    if policy_scores.ndim != 2 or reward_scores.ndim != 2:
        raise ValueError("policy_scores and reward_scores must have shape [batch, K]")
    if policy_scores.shape != reward_scores.shape:
        raise ValueError(
            f"policy_scores and reward_scores must have same shape, got "
            f"{tuple(policy_scores.shape)} and {tuple(reward_scores.shape)}"
        )
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    p = torch.softmax(policy_scores / temperature, dim=-1)
    if center_rewards:
        baseline = (p * reward_scores).sum(dim=-1, keepdim=True)
        centered_reward = reward_scores - baseline
    else:
        centered_reward = reward_scores

    q = torch.softmax((policy_scores + eta * centered_reward) / temperature, dim=-1)

    log_p = torch.log(p.clamp_min(eps))
    log_q = torch.log(q.clamp_min(eps))
    kl_q_p = (q * (log_q - log_p)).sum(dim=-1)
    target_entropy = -(q * log_q).sum(dim=-1)
    policy_entropy = -(p * log_p).sum(dim=-1)
    reward_gain = ((q - p) * reward_scores).sum(dim=-1)

    stats = DDORMStats(
        mean_kl_q_p=kl_q_p.mean(),
        mean_target_entropy=target_entropy.mean(),
        mean_policy_entropy=policy_entropy.mean(),
        mean_reward_gain_under_rm=reward_gain.mean(),
        mean_abs_centered_reward=centered_reward.abs().mean(),
    )
    return p, q, centered_reward, stats


def ddorm_cross_entropy_loss(
    policy_scores: torch.Tensor,
    target_q: torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Cross-entropy CE(q, p_theta) over finite candidates."""
    logp = F.log_softmax(policy_scores / temperature, dim=-1)
    return -(target_q * logp).sum(dim=-1).mean()


@torch.no_grad()
def effective_kl_curve(
    policy_scores: torch.Tensor,
    reward_scores: torch.Tensor,
    betas: torch.Tensor,
    *,
    center_rewards: bool = True,
) -> torch.Tensor:
    """Return mean KL(q_beta || p) for beta = eta / temperature.

    This helper is useful for tuning a PPO-like trust-region radius before a run.
    It uses temperature=1 and eta=beta, which is equivalent to changing eta/tau.
    """
    out = []
    for beta in betas.detach().cpu().tolist():
        _, _, _, stats = ddorm_target_distribution(
            policy_scores,
            reward_scores,
            eta=float(beta),
            temperature=1.0,
            center_rewards=center_rewards,
        )
        out.append(stats.mean_kl_q_p)
    return torch.stack(out)
