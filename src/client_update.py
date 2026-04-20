from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch import amp
from torch.utils.data import DataLoader


def train_one_client(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    use_amp: bool,
    max_batches: Optional[int] = None,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0
    total_n = 0
    scaler: Optional[amp.GradScaler] = None
    if bool(use_amp) and device.type == "cuda":
        scaler = amp.GradScaler("cuda")

    batches_seen = 0
    for _ in range(int(epochs)):
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if scaler is not None:
                with amp.autocast("cuda"):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            bs = int(x.size(0))
            total_loss += float(loss.detach().cpu()) * bs
            total_n += bs

            batches_seen += 1
            if max_batches is not None and batches_seen >= int(max_batches):
                avg_loss = total_loss / max(total_n, 1)
                return avg_loss, total_n

    avg_loss = total_loss / max(total_n, 1)
    return avg_loss, total_n
