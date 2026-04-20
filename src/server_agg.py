from __future__ import annotations

from typing import Dict, List, Sequence

import torch


def fedavg_weighted(state_dicts: List[Dict[str, torch.Tensor]], weights: Sequence[float]) -> Dict[str, torch.Tensor]:
    if not state_dicts:
        raise ValueError("state_dicts is empty")
    if len(state_dicts) != len(weights):
        raise ValueError("state_dicts and weights length mismatch")

    wsum = float(sum(float(w) for w in weights))
    if wsum <= 0:
        raise ValueError("Sum of weights must be > 0")

    keys = list(state_dicts[0].keys())
    for sd in state_dicts[1:]:
        if list(sd.keys()) != keys:
            raise ValueError("Inconsistent state_dict keys across clients")

    out: Dict[str, torch.Tensor] = {}
    for k in keys:
        ref = state_dicts[0][k]
        acc = torch.zeros_like(ref, dtype=torch.float32)
        for sd, w in zip(state_dicts, weights):
            acc += sd[k].float() * (float(w) / wsum)
        out[k] = acc.to(dtype=ref.dtype)
    return out
