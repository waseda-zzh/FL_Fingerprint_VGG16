from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch


def save_client_upload_bundle(
    base_dir: Path,
    *,
    client_id: int,
    round_idx: int,
    upload_index: int,
    state_dict: Mapping[str, torch.Tensor],
    meta: dict[str, Any],
) -> Path:
    """Persist one client's upload for one participation in a FL round.

    Layout (each client has its own top-level folder):

      base_dir/client_{id}/round_{r}/upload_{u}/state_dict.pt
      base_dir/client_{id}/round_{r}/upload_{u}/meta.json

    ``upload_index`` disambiguates multiple participations of the same client in one round
    (possible when client sampling uses replacement).
    """
    out_dir = base_dir / f"client_{int(client_id):03d}" / f"round_{int(round_idx):04d}" / f"upload_{int(upload_index):02d}"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(dict(state_dict), out_dir / "state_dict.pt")
    payload = dict(meta)
    payload["saved_at_utc"] = datetime.now(timezone.utc).isoformat()
    (out_dir / "meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir
