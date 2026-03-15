from __future__ import annotations

import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .utils import format_pm

try:
    from IPython.display import display
except Exception:
    display = None


def summarize_records(records: List[Dict[str, float]], group_keys: Tuple[str, ...]) -> List[Dict[str, float]]:
    buckets: Dict[Tuple[object, ...], List[Dict[str, float]]] = {}
    for row in records:
        key = tuple(row[k] for k in group_keys)
        buckets.setdefault(key, []).append(row)

    out: List[Dict[str, float]] = []
    for key, rows in sorted(buckets.items(), key=lambda kv: kv[0]):
        ps = np.array([r["PSNR"] for r in rows], dtype=float)
        ss = np.array([r["SSIM"] for r in rows], dtype=float)
        nr = np.array([r["NRMSE"] for r in rows], dtype=float)

        item: Dict[str, float] = {}
        for name, value in zip(group_keys, key):
            item[name] = value

        item.update(
            {
                "PSNR_mean": float(ps.mean()),
                "PSNR_std": float(ps.std(ddof=1) if len(ps) > 1 else 0.0),
                "SSIM_mean": float(ss.mean()),
                "SSIM_std": float(ss.std(ddof=1) if len(ss) > 1 else 0.0),
                "NRMSE_mean": float(nr.mean()),
                "NRMSE_std": float(nr.std(ddof=1) if len(nr) > 1 else 0.0),
                "n": int(len(rows)),
            }
        )
        out.append(item)
    return out


def add_pretty_columns(rows: List[Dict[str, float]], digits: int = 4) -> List[Dict[str, float]]:
    out = []
    for row in rows:
        row2 = dict(row)
        row2["PSNR_mean±std"] = format_pm(float(row["PSNR_mean"]), float(row["PSNR_std"]), digits)
        row2["SSIM_mean±std"] = format_pm(float(row["SSIM_mean"]), float(row["SSIM_std"]), digits)
        row2["NRMSE_mean±std"] = format_pm(float(row["NRMSE_mean"]), float(row["NRMSE_std"]), digits)
        out.append(row2)
    return out


def save_csv(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_markdown_table(path: str, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write(pd.DataFrame(rows).to_markdown(index=False))


def method_order() -> List[str]:
    return [
        "cg_sense_tikh",
        "fista_sense_wavelet",
        "fista_sense_tfft",
        "pgd_sense_lowrank",
        "prop2_sense_pgd_v2",
    ]


def sort_summary_rows(rows: List[Dict[str, float]]) -> List[Dict[str, float]]:
    order = {m: i for i, m in enumerate(method_order())}
    return sorted(
        rows,
        key=lambda row: (
            row.get("file", ""),
            int(row.get("R", 0)),
            order.get(str(row.get("method", "")), 999),
            str(row.get("method", "")),
        ),
    )


def render_compare_figure(ref: np.ndarray, preds: Dict[str, np.ndarray], title: str, path: str, show: bool = False) -> None:
    methods = method_order()
    items = [("reference", ref)] + [(m, preds[m]) for m in methods if m in preds]
    n = len(items)

    plt.figure(figsize=(3 * n, 6))
    for i, (name, img) in enumerate(items, start=1):
        plt.subplot(2, n, i)
        plt.imshow(np.abs(img), cmap="gray")
        plt.axis("off")
        plt.title(name)

        if name != "reference":
            plt.subplot(2, n, i + n)
            plt.imshow(np.abs(ref - img), cmap="magma")
            plt.axis("off")
            plt.title(f"|Ref-{name}|")

    plt.suptitle(title)
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=160, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


def manuscript_ready_summary(summary_rows: List[Dict[str, float]]) -> str:
    lines = []
    by_r: Dict[int, List[Dict[str, float]]] = {}
    for row in summary_rows:
        by_r.setdefault(int(row["R"]), []).append(row)

    for R in sorted(by_r):
        rows = by_r[R]
        best_psnr = max(rows, key=lambda z: z["PSNR_mean"])
        best_ssim = max(rows, key=lambda z: z["SSIM_mean"])
        best_nrmse = min(rows, key=lambda z: z["NRMSE_mean"])
        lines.append(
            f"R={R}: tốt nhất theo PSNR là {best_psnr['method']} ({best_psnr['PSNR_mean']:.4f}±{best_psnr['PSNR_std']:.4f}), "
            f"tốt nhất theo SSIM là {best_ssim['method']} ({best_ssim['SSIM_mean']:.4f}±{best_ssim['SSIM_std']:.4f}), "
            f"và tốt nhất theo NRMSE là {best_nrmse['method']} ({best_nrmse['NRMSE_mean']:.4f}±{best_nrmse['NRMSE_std']:.4f})."
        )
    return "\n".join(lines)


def display_summary_tables(out_dir: str) -> None:
    try:
        overall_df = pd.read_csv(os.path.join(out_dir, "summary_overall_pretty.csv"))
        by_file_df = pd.read_csv(os.path.join(out_dir, "summary_by_file_pretty.csv"))

        print("\n=== summary_overall_pretty.csv ===")
        if display is not None:
            display(overall_df)
        else:
            print(overall_df.to_string(index=False))

        print("\n=== summary_by_file_pretty.csv ===")
        if display is not None:
            display(by_file_df)
        else:
            print(by_file_df.to_string(index=False))
    except Exception as exc:
        print(f"Could not display summary tables: {exc}")
