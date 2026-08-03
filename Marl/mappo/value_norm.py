"""
value_norm.py

Value Normalization used in the official MAPPO implementation.

Maintains running statistics of the value targets
and normalizes critic regression targets.

Reference:
Official MAPPO implementation (marlbenchmark/on-policy)
"""

import torch
import torch.nn as nn


class ValueNorm(nn.Module):
    """
    Running mean / variance normalizer for value targets.
    """

    def __init__(
        self,
        input_shape=1,
        beta=0.99999,
        epsilon=1e-5,
        device="cpu",
    ):
        super().__init__()

        self.input_shape = input_shape
        self.beta = beta
        self.epsilon = epsilon

        self.running_mean = nn.Parameter(
            torch.zeros(input_shape, device=device),
            requires_grad=False,
        )

        self.running_mean_sq = nn.Parameter(
            torch.zeros(input_shape, device=device),
            requires_grad=False,
        )

        self.debiasing_term = nn.Parameter(
            torch.tensor(0.0, device=device),
            requires_grad=False,
        )

    @torch.no_grad()
    def update(self, values):
        """
        Update running statistics using a batch of returns.

        values:
            Tensor of shape (N,) or (N,1)
        """

        values = values.float()

        batch_mean = values.mean(dim=0)
        batch_sq_mean = (values ** 2).mean(dim=0)

        self.running_mean.mul_(self.beta).add_(
            batch_mean * (1.0 - self.beta)
        )

        self.running_mean_sq.mul_(self.beta).add_(
            batch_sq_mean * (1.0 - self.beta)
        )

        self.debiasing_term.mul_(self.beta).add_(
            torch.tensor(
                1.0 - self.beta,
                device=values.device,
            )
        )

    def running_mean_var(self):
        """
        Returns debiased mean and variance.
        """

        debias = torch.clamp(
            self.debiasing_term,
            min=self.epsilon,
        )

        mean = self.running_mean / debias

        mean_sq = self.running_mean_sq / debias

        var = torch.clamp(
            mean_sq - mean ** 2,
            min=1e-2,
        )

        return mean, var

    def normalize(self, values):
        """
        Normalize value targets.
        """

        mean, var = self.running_mean_var()

        return (values - mean) / torch.sqrt(var)

    def denormalize(self, values):
        """
        Convert normalized predictions
        back to the original reward scale.
        """

        mean, var = self.running_mean_var()

        return values * torch.sqrt(var) + mean