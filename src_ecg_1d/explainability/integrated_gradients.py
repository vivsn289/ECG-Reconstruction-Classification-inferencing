# src_ecg_1d/explainability/integrated_gradients.py
#
# Integrated Gradients (IG) implementation for ECG attribution.
#
# Reference: Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
#            https://arxiv.org/abs/1703.01365
#
# IG approximates the path integral of gradients from a baseline input
# to the actual input. The resulting attribution tensor has the same
# shape as the input and satisfies the Completeness axiom:
#   sum(IG) = model(input) - model(baseline)
#
# For ECG:
# - Input shape:    (1, channels, time) = (1, 12, 1000)
# - Baseline shape: same as input
# - Attribution:    same shape as input — one value per (lead, timestep)

import numpy as np
import torch


def integrated_gradients(
    model,
    x: torch.Tensor,
    target_class: int,
    baseline: torch.Tensor = None,
    steps: int = 50,
) -> torch.Tensor:
    """Compute Integrated Gradients attributions for a single input.

    Args:
        model: PyTorch model in eval mode. Must return a (1, num_classes)
               logit tensor when called with a (1, channels, time) input.
        x: input tensor of shape (1, channels, time)
        target_class: integer class index whose score is explained
        baseline: baseline tensor, same shape as x.
            Defaults to an all-zeros tensor if not provided.
            Common choices: zeros, noise average, per-channel mean.
        steps: number of Riemann approximation steps (more → more accurate)

    Returns:
        attributions: tensor of shape (channels, time).
            Each value represents how much that (lead, timestep) contributed
            to the model's output for target_class relative to the baseline.

    Notes:
        - We skip α=0 (baseline) since its gradient contributes trivially.
        - Gradients are accumulated and averaged (Riemann approximation).
        - The model must support gradient flow through its forward pass.
          Ensure it is NOT called inside torch.no_grad() before this.
    """
    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(x)

    assert x.shape == baseline.shape, (
        f"Input shape {x.shape} must match baseline shape {baseline.shape}"
    )

    total_grad = torch.zeros_like(x)

    # Interpolate from baseline+epsilon to input in `steps` steps.
    # We skip alpha=0 (pure baseline) since its gradient is trivial.
    for alpha in np.linspace(0, 1, steps, endpoint=False)[1:]:
        x_step = baseline + float(alpha) * (x - baseline)
        x_step = x_step.clone().detach().requires_grad_(True)

        logits = model(x_step)
        score = logits[0, target_class]

        model.zero_grad()
        score.backward()
        total_grad += x_step.grad.detach()

    avg_grad = total_grad / steps

    # IG formula: (input − baseline) × average_gradient
    attributions = (x - baseline) * avg_grad

    # Return shape: (channels, time) — drop the batch dimension
    return attributions.squeeze(0)
