#!/usr/bin/env python3
"""
分析 client_uploads/ 下各客户端跨轮次上传的 state_dict 变化（与 FL_watermark 中
``experiments/analyze_client_upload_drift.py`` **相同的输出 CSV 格式与指标**；目录
发现适配本仓库布局）。

**内存**：对每个客户端只在内存中保留 **首轮快照 + 当前相邻两轮**（最多约 3 份
整模）；每做完一步相邻轮比较就把该步结果落盘（per_round 行追加；param/neuron
各一列写入临时 .npy），再加载下一轮。最终宽表由 mmap 按行写出，避免在内存里堆
整条时间链上的所有 state_dict。

本仓库布局::

    <client_uploads>/client_XXX/round_RRRR/upload_UU/state_dict.pt

输出（默认在 client_uploads 父目录）:
  - client_upload_drift_report.csv
  - client_upload_drift_per_round.csv
  - client_upload_drift_matrices/client_XXXX_param_drift.csv
  - client_upload_neuron_matrices/client_XXXX_neuron_drift.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch

TimelineEntry = Tuple[int, str, Optional[float]]


def _load_state(path: str) -> Dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _state_dict_to_vector(sd: Dict[str, Any]) -> torch.Tensor:
    keys = sorted(sd.keys())
    parts: List[torch.Tensor] = []
    for k in keys:
        t = sd[k]
        if not torch.is_tensor(t):
            continue
        parts.append(t.detach().float().reshape(-1))
    if not parts:
        return torch.tensor([])
    return torch.cat(parts)


def _l2_between(sd_a: Dict[str, Any], sd_b: Dict[str, Any]) -> float:
    if set(sd_a.keys()) != set(sd_b.keys()):
        raise ValueError(f"state_dict keys mismatch: {set(sd_a.keys()) ^ set(sd_b.keys())}")
    va = _state_dict_to_vector(sd_a)
    vb = _state_dict_to_vector(sd_b)
    return float(torch.norm(vb - va, p=2).item())


def _average_state_dict_paths(paths: List[str]) -> Dict[str, Any]:
    if not paths:
        raise ValueError("empty paths")
    acc = _load_state(paths[0])
    acc = {k: v.detach().float().clone() for k, v in acc.items() if torch.is_tensor(v)}
    for p in paths[1:]:
        sd = _load_state(p)
        sd = {k: v for k, v in sd.items() if torch.is_tensor(v)}
        if set(sd.keys()) != set(acc.keys()):
            raise ValueError(f"State dict keys mismatch: {paths[0]} vs {p}")
        for k in acc:
            acc[k] = acc[k] + sd[k]
    for k in acc:
        acc[k] = acc[k] / float(len(paths))
    return acc


def _avg_train_loss_for_round(round_dir: Path) -> Optional[float]:
    vals: List[float] = []
    for up in sorted(round_dir.glob("upload_*")):
        if not up.is_dir():
            continue
        meta = up / "meta.json"
        if not meta.is_file():
            continue
        try:
            obj = json.loads(meta.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if obj.get("avg_local_loss") is not None:
            vals.append(float(obj["avg_local_loss"]))
    if not vals:
        return None
    return float(sum(vals) / len(vals))


def _build_blackbox_timeline(upload_root: Path) -> DefaultDict[int, List[TimelineEntry]]:
    timeline: DefaultDict[int, List[TimelineEntry]] = defaultdict(list)
    if not upload_root.is_dir():
        raise FileNotFoundError(f"Not a directory: {upload_root}")

    client_pat = re.compile(r"^client_(\d+)$", re.IGNORECASE)
    round_pat = re.compile(r"^round_(\d+)$", re.IGNORECASE)

    for client_dir in sorted(upload_root.iterdir()):
        if not client_dir.is_dir():
            continue
        m = client_pat.match(client_dir.name)
        if not m:
            continue
        cid = int(m.group(1))
        for round_dir in sorted(client_dir.iterdir()):
            if not round_dir.is_dir():
                continue
            m2 = round_pat.match(round_dir.name)
            if not m2:
                continue
            r = int(m2.group(1))
            pts = sorted(round_dir.glob("upload_*/state_dict.pt"))
            if not pts:
                continue
            paths = [str(p) for p in pts]
            loss = _avg_train_loss_for_round(round_dir)
            timeline[cid].append((r, paths[0], loss))

    for cid in timeline:
        timeline[cid].sort(key=lambda x: x[0])
    return timeline


def _round_paths_from_entry(path0: str) -> List[str]:
    round_dir = Path(path0).parent.parent
    pts = sorted(round_dir.glob("upload_*/state_dict.pt"))
    return [str(p) for p in pts]


def _neuron_rows_for_param(delta: torch.Tensor, name: str) -> List[Tuple[str, float]]:
    x = delta.detach().float()
    rows: List[Tuple[str, float]] = []
    if x.dim() == 0:
        rows.append((f"{name}:scalar", float(x.abs().item())))
        return rows
    if x.dim() == 1:
        for i in range(x.shape[0]):
            rows.append((f"{name}:n{i}", float(x[i].abs().item())))
        return rows

    x2 = x.reshape(x.shape[0], -1)
    for i in range(x2.shape[0]):
        rows.append((f"{name}:n{i}", float(torch.norm(x2[i], p=2).item())))
    return rows


def _neuron_schema_from_first_pair(sd0: Dict[str, Any], sd1: Dict[str, Any]) -> Tuple[List[str], Dict[str, int]]:
    keys = sorted(sd0.keys())
    first_map: Dict[str, float] = {}
    for k in keys:
        d = sd1[k].detach().float() - sd0[k].detach().float()
        for rn, rv in _neuron_rows_for_param(d, k):
            first_map[rn] = rv
    row_names = sorted(first_map.keys())
    row_index = {rn: i for i, rn in enumerate(row_names)}
    return row_names, row_index


def _build_param_column(prev_sd: Dict[str, Any], next_sd: Dict[str, Any], keys: List[str]) -> np.ndarray:
    col = np.zeros(len(keys), dtype=np.float32)
    for i, k in enumerate(keys):
        a = prev_sd[k].detach().float().reshape(-1)
        b = next_sd[k].detach().float().reshape(-1)
        col[i] = float(torch.norm(b - a, p=2).item())
    return col


def _build_neuron_column(
    prev_sd: Dict[str, Any],
    next_sd: Dict[str, Any],
    keys: List[str],
    row_index: Dict[str, int],
    n_rows: int,
) -> np.ndarray:
    v = np.zeros(n_rows, dtype=np.float32)
    for k in keys:
        d = next_sd[k].detach().float() - prev_sd[k].detach().float()
        for rn, rv in _neuron_rows_for_param(d, k):
            j = row_index.get(rn)
            if j is not None:
                v[j] = rv
    return v


def _build_neuron_relative_column(
    prev_sd: Dict[str, Any],
    next_sd: Dict[str, Any],
    keys: List[str],
    row_index: Dict[str, int],
    n_rows: int,
    eps: float,
) -> np.ndarray:
    """Per-neuron relative drift: ||delta|| / (||prev|| + eps)."""
    v = np.zeros(n_rows, dtype=np.float32)
    for k in keys:
        prev = prev_sd[k].detach().float()
        nxt = next_sd[k].detach().float()
        delta = nxt - prev
        if delta.dim() == 0:
            rn = f"{k}:scalar"
            j = row_index.get(rn)
            if j is not None:
                num = float(delta.abs().item())
                den = float(prev.abs().item()) + eps
                v[j] = num / den
            continue
        if delta.dim() == 1:
            for i in range(delta.shape[0]):
                rn = f"{k}:n{i}"
                j = row_index.get(rn)
                if j is not None:
                    num = float(delta[i].abs().item())
                    den = float(prev[i].abs().item()) + eps
                    v[j] = num / den
            continue

        d2 = delta.reshape(delta.shape[0], -1)
        p2 = prev.reshape(prev.shape[0], -1)
        for i in range(d2.shape[0]):
            rn = f"{k}:n{i}"
            j = row_index.get(rn)
            if j is not None:
                num = float(torch.norm(d2[i], p=2).item())
                den = float(torch.norm(p2[i], p=2).item()) + eps
                v[j] = num / den
    return v


def _write_wide_from_column_npys(
    out_path: str,
    row_names: List[str],
    col_labels: List[str],
    col_paths: List[str],
    full_l2_per_step: Optional[List[float]],
    first_col_header: str,
) -> None:
    d = os.path.dirname(out_path)
    if d:
        os.makedirs(d, exist_ok=True)
    mmaps: List[Any] = [np.load(p, mmap_mode="r") for p in col_paths]
    try:
        n_rows = len(row_names)
        for m in mmaps:
            if int(m.shape[0]) != n_rows:
                raise ValueError(f"Column shape {m.shape} != n_rows {n_rows}")
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([first_col_header, *col_labels])
            for i, name in enumerate(row_names):
                row = [name] + [f"{float(mmaps[t][i]):.8f}" for t in range(len(mmaps))]
                w.writerow(row)
            if full_l2_per_step is not None:
                w.writerow(["<<FULL_MODEL_L2>>", *[f"{v:.8f}" for v in full_l2_per_step]])
    finally:
        del mmaps


def _process_client_streaming(
    cid: int,
    entries: List[TimelineEntry],
    *,
    per_round_writer: csv.DictWriter,
    per_round_file,
    matrix_dir: str,
    neuron_dir: str,
    neuron_rel_dir: str,
    write_param: bool,
    write_neuron: bool,
    relative_eps: float,
) -> Dict[str, Any]:
    """Stream one client: append per-round rows; write column npys then assemble matrices."""
    n = len(entries)
    if n == 0:
        return {
            "num_snapshots": 0,
            "num_steps": 0,
            "mean_l2_step": float("nan"),
            "std_l2_step": float("nan"),
            "max_l2_step": float("nan"),
            "total_l2_path": float("nan"),
            "l2_first_to_last": float("nan"),
            "first_round": None,
            "last_round": None,
            "client_id": cid,
        }
    if n == 1:
        r0, _, _ = entries[0]
        return {
            "num_snapshots": 1,
            "num_steps": 0,
            "mean_l2_step": float("nan"),
            "std_l2_step": float("nan"),
            "max_l2_step": float("nan"),
            "total_l2_path": 0.0,
            "l2_first_to_last": 0.0,
            "first_round": r0,
            "last_round": r0,
            "client_id": cid,
        }

    if write_param:
        tmp_parent = matrix_dir
    elif write_neuron:
        tmp_parent = neuron_dir
    else:
        tmp_parent = None
    if tmp_parent:
        os.makedirs(tmp_parent, exist_ok=True)
    tmp_root = tempfile.mkdtemp(prefix=f"drift_c{cid}_", dir=tmp_parent)

    steps: List[float] = []
    col_labels: List[str] = []
    param_col_files: List[str] = []
    neuron_col_files: List[str] = []
    neuron_rel_col_files: List[str] = []
    row_names: List[str] = []
    row_index: Dict[str, int] = {}
    keys: List[str] = []
    l2_fl = 0.0

    try:
        first_sd = _average_state_dict_paths(_round_paths_from_entry(entries[0][1]))
        keys = sorted(first_sd.keys())

        if write_param or write_neuron:
            os.makedirs(matrix_dir, exist_ok=True)
        if write_neuron:
            os.makedirs(neuron_dir, exist_ok=True)
            os.makedirs(neuron_rel_dir, exist_ok=True)

        sd_next = _average_state_dict_paths(_round_paths_from_entry(entries[1][1]))
        if write_neuron:
            row_names, row_index = _neuron_schema_from_first_pair(first_sd, sd_next)

        prev_sd = first_sd
        cum = 0.0
        for i in range(n - 1):
            r0, _, l0 = entries[i]
            r1, _, l1 = entries[i + 1]
            if i > 0:
                sd_next = _average_state_dict_paths(_round_paths_from_entry(entries[i + 1][1]))

            d_step = _l2_between(prev_sd, sd_next)
            steps.append(d_step)
            cum += d_step
            d_from_first = _l2_between(first_sd, sd_next)

            col_lbl = f"r{r0}_to_r{r1}"
            col_labels.append(col_lbl)

            if write_param:
                pcol = _build_param_column(prev_sd, sd_next, keys)
                pf = os.path.join(tmp_root, f"param_step_{i:04d}.npy")
                np.save(pf, pcol)
                param_col_files.append(pf)

            if write_neuron and row_names:
                ncol = _build_neuron_column(prev_sd, sd_next, keys, row_index, len(row_names))
                nf = os.path.join(tmp_root, f"neuron_step_{i:04d}.npy")
                np.save(nf, ncol)
                neuron_col_files.append(nf)
                nrcol = _build_neuron_relative_column(
                    prev_sd, sd_next, keys, row_index, len(row_names), relative_eps
                )
                nrf = os.path.join(tmp_root, f"neuron_rel_step_{i:04d}.npy")
                np.save(nrf, nrcol)
                neuron_rel_col_files.append(nrf)

            per_round_writer.writerow(
                {
                    "client_id": cid,
                    "step_index": i,
                    "round_from": r0,
                    "round_to": r1,
                    "round_display_from": r0 + 1,
                    "round_display_to": r1 + 1,
                    "l2_step": d_step,
                    "cumulative_l2_path": cum,
                    "l2_from_first_snapshot": d_from_first,
                    "train_loss_at_round_from": "" if l0 is None else f"{l0:.8f}",
                    "train_loss_at_round_to": "" if l1 is None else f"{l1:.8f}",
                }
            )
            per_round_file.flush()

            if prev_sd is not first_sd:
                del prev_sd
            prev_sd = sd_next

        del first_sd
        if prev_sd is not None:
            del prev_sd
        del sd_next

        sd_first = _average_state_dict_paths(_round_paths_from_entry(entries[0][1]))
        sd_last = _average_state_dict_paths(_round_paths_from_entry(entries[-1][1]))
        l2_fl = _l2_between(sd_first, sd_last)
        del sd_first, sd_last

        if write_param and param_col_files:
            mpath = os.path.join(matrix_dir, f"client_{cid:04d}_param_drift.csv")
            _write_wide_from_column_npys(
                mpath, keys, col_labels, param_col_files, steps, "param_name"
            )
        if write_neuron and neuron_col_files and row_names:
            npath = os.path.join(neuron_dir, f"client_{cid:04d}_neuron_drift.csv")
            _write_wide_from_column_npys(
                npath, row_names, col_labels, neuron_col_files, steps, "neuron_name"
            )
        if write_neuron and neuron_rel_col_files and row_names:
            nrpath = os.path.join(neuron_rel_dir, f"client_{cid:04d}_neuron_rel_drift.csv")
            _write_wide_from_column_npys(
                nrpath, row_names, col_labels, neuron_rel_col_files, None, "neuron_name"
            )
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    agg = {
        "client_id": cid,
        "num_snapshots": n,
        "num_steps": len(steps),
        "mean_l2_step": float(statistics.mean(steps)) if steps else float("nan"),
        "std_l2_step": float(statistics.pstdev(steps)) if len(steps) > 1 else (0.0 if steps else float("nan")),
        "max_l2_step": max(steps) if steps else float("nan"),
        "total_l2_path": float(sum(steps)) if steps else float("nan"),
        "l2_first_to_last": l2_fl,
        "first_round": entries[0][0],
        "last_round": entries[-1][0],
    }
    return agg


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_uploads = repo_root / "artifacts" / "client_uploads"

    p = argparse.ArgumentParser(
        description="Streaming drift analysis (FL_watermark-compatible CSV outputs)."
    )
    p.add_argument(
        "client_uploads_dir",
        nargs="?",
        default=str(default_uploads),
        help=f"Path to client_uploads (default: {default_uploads})",
    )
    p.add_argument("--out_csv", type=str, default="")
    p.add_argument("--out_per_round_csv", type=str, default="")
    p.add_argument("--stagnant_frac", type=float, default=0.1)
    p.add_argument("--matrix_dir", type=str, default="")
    p.add_argument("--no_param_matrices", action="store_true")
    p.add_argument("--neuron_matrix_dir", type=str, default="")
    p.add_argument("--neuron_rel_matrix_dir", type=str, default="")
    p.add_argument("--no_neuron_matrices", action="store_true")
    p.add_argument("--relative_eps", type=float, default=1e-12)
    args = p.parse_args()

    upload_root = Path(os.path.abspath(args.client_uploads_dir))
    timeline = _build_blackbox_timeline(upload_root)
    client_ids = sorted(timeline.keys())
    if not client_ids:
        print(f"No client_* / round_* / state_dict.pt under {upload_root}", file=sys.stderr)
        sys.exit(1)

    parent = str(upload_root.parent)
    out_csv = args.out_csv or os.path.join(parent, "client_upload_drift_report.csv")
    out_pr = args.out_per_round_csv or os.path.join(parent, "client_upload_drift_per_round.csv")
    matrix_dir = args.matrix_dir or os.path.join(parent, "client_upload_drift_matrices")
    neuron_dir = args.neuron_matrix_dir or os.path.join(parent, "client_upload_neuron_matrices")
    neuron_rel_dir = args.neuron_rel_matrix_dir or os.path.join(
        parent, "client_upload_neuron_rel_matrices"
    )

    per_round_fields = [
        "client_id",
        "step_index",
        "round_from",
        "round_to",
        "round_display_from",
        "round_display_to",
        "l2_step",
        "cumulative_l2_path",
        "l2_from_first_snapshot",
        "train_loss_at_round_from",
        "train_loss_at_round_to",
    ]

    rows: List[Dict[str, Any]] = []

    with open(out_pr, "w", newline="", encoding="utf-8") as f_pr:
        w_pr = csv.DictWriter(f_pr, fieldnames=per_round_fields, extrasaction="ignore")
        w_pr.writeheader()
        f_pr.flush()

        for cid in client_ids:
            ent = timeline[cid]
            agg = _process_client_streaming(
                cid,
                ent,
                per_round_writer=w_pr,
                per_round_file=f_pr,
                matrix_dir=matrix_dir,
                neuron_dir=neuron_dir,
                neuron_rel_dir=neuron_rel_dir,
                write_param=not args.no_param_matrices,
                write_neuron=not args.no_neuron_matrices,
                relative_eps=args.relative_eps,
            )
            rows.append(agg)
            f_pr.flush()

    valid_for_rank = [r for r in rows if r["num_steps"] > 0]
    for r in rows:
        if r["num_steps"] == 0:
            r["drift_label"] = "single_or_no_step"
    sorted_by_mean = sorted(valid_for_rank, key=lambda x: x["mean_l2_step"])
    k_stag = max(1, int(len(sorted_by_mean) * args.stagnant_frac)) if sorted_by_mean else 0
    threshold_low = sorted_by_mean[k_stag - 1]["mean_l2_step"] if k_stag else float("nan")
    for i, r in enumerate(sorted_by_mean):
        r["drift_label"] = "low_drift" if i < k_stag else "normal"

    fieldnames = [
        "client_id",
        "first_round",
        "last_round",
        "num_snapshots",
        "num_steps",
        "mean_l2_step",
        "std_l2_step",
        "max_l2_step",
        "total_l2_path",
        "l2_first_to_last",
        "drift_label",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in sorted(rows, key=lambda x: x["client_id"]):
            w.writerow({k: r.get(k, "") for k in fieldnames})

    print(f"\nWrote summary:  {out_csv}")
    print(f"Wrote per-step: {out_pr} (streaming rows)")
    if not args.no_param_matrices:
        print(f"Wrote matrices: {matrix_dir}/ (client_XXXX_param_drift.csv)")
    if not args.no_neuron_matrices:
        print(f"Wrote neuron matrices: {neuron_dir}/ (client_XXXX_neuron_drift.csv)")
        print(
            f"Wrote neuron relative matrices: {neuron_rel_dir}/ "
            "(client_XXXX_neuron_rel_drift.csv)"
        )

    print("=" * 72)
    for r in sorted(rows, key=lambda x: x["client_id"]):
        cid = r["client_id"]
        if r["num_steps"] == 0:
            print(
                f"  client {cid:3d}: snapshots={r['num_snapshots']}, "
                f"rounds {r['first_round']}–{r['last_round']} — no consecutive pair"
            )
            continue
        print(
            f"  client {cid:3d}: steps={r['num_steps']}, rounds {r['first_round']}–{r['last_round']} | "
            f"mean_step={r['mean_l2_step']:.4f}, max_step={r['max_l2_step']:.4f}, "
            f"path_sum={r['total_l2_path']:.4f}, ||last-first||={r['l2_first_to_last']:.4f} | {r['drift_label']}"
        )
    if sorted_by_mean:
        print("\n" + "=" * 72)
        print(
            f"Low-drift: smallest {k_stag} client(s) by mean_l2_step "
            f"(~{args.stagnant_frac * 100:.0f}% of clients with ≥2 snapshots). "
            f"Largest mean among that set: {threshold_low:.6f}"
        )
        low = [r["client_id"] for r in rows if r.get("drift_label") == "low_drift"]
        if low:
            print(f"Clients tagged low_drift: {sorted(low)}")
    print("=" * 72)


if __name__ == "__main__":
    main()
