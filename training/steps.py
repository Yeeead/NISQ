from __future__ import annotations

import torch


def optimizer_step(loss, optimizer, grad_clip=None) -> None:
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip is not None:
        torch.nn.utils.clip_grad_norm_(
            [param for group in optimizer.param_groups for param in group["params"]],
            float(grad_clip),
        )
    optimizer.step()
