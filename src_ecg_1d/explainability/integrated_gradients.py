# src_ecg_1d/explainability/integrated_gradients.py

import torch


def integrated_gradients(
    model,
    x,
    target_class,
    baseline=None,
    steps=50,
):

    model.eval()

    if baseline is None:
        baseline = torch.zeros_like(x)

    scaled_inputs = [
        baseline + (float(i) / steps) * (x - baseline)
        for i in range(steps + 1)
    ]

    grads = []

    for inp in scaled_inputs:
        inp.requires_grad_(True)
        logits = model(inp)
        score = logits[0, target_class]
        score.backward()
        grads.append(inp.grad.detach())

    avg_grads = torch.mean(torch.stack(grads), dim=0)
    attributions = (x - baseline) * avg_grads

    return attributions.squeeze(0)
