from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Optional


def append_metrics_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_round_metrics_csv(
    path: Path,
    *,
    round_idx: int,
    train_loss_weighted: float,
    eval_top1: Optional[float],
    eval_top5: Optional[float],
    num_participants: int,
) -> None:
    """One row per round; header written on first create."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "round",
        "train_loss_weighted",
        "eval_top1",
        "eval_top5",
        "num_participants",
    ]
    row = {
        "round": int(round_idx),
        "train_loss_weighted": float(train_loss_weighted),
        "eval_top1": "" if eval_top1 is None else float(eval_top1),
        "eval_top5": "" if eval_top5 is None else float(eval_top5),
        "num_participants": int(num_participants),
    }
    new_file = not path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new_file:
            w.writeheader()
        w.writerow(row)
