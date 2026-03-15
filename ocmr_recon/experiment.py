from __future__ import annotations

import json
import os
from copy import deepcopy
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import ExperimentConfig, RunConfig
from .data import read_ocmr_kspace
from .methods.baselines import BaselineConfig, cg_sense_tikh, fista_tfft, fista_wavelet, pgd_lowrank
from .methods.proposed import prop2_sense_pgd_improved
from .ops import (
    SenseForwardOp,
    build_vd_mask,
    estimate_lipschitz,
    estimate_sens_maps_from_acs,
    make_reference_from_full_kspace,
    make_reference_sense,
)
from .reporting import add_pretty_columns, render_compare_figure, save_csv, save_markdown_table, sort_summary_rows, summarize_records
from .utils import add_complex_noise, center_crop, compute_metrics


def run_single_case(
    full_kspace: np.ndarray,
    ref: np.ndarray,
    sens: np.ndarray,
    file_stem: str,
    R: int,
    seed: int,
    cfg: ExperimentConfig,
    ky_center: Optional[int],
) -> Tuple[List[Dict[str, float]], Dict[str, np.ndarray], List[Dict[str, float]]]:
    _, _, ny, nx = full_kspace.shape
    mask = build_vd_mask(ny, nx, R=R, acs_lines=cfg.acs_lines, seed=seed, ky_center=ky_center)
    op = SenseForwardOp(sens, mask)

    y = mask[None, None, :, :] * full_kspace
    y = add_complex_noise(y, mask, cfg.noise_std, seed=seed + 123)

    ref_frame = ref[len(ref) // 2]
    print(
        f"  Seed={seed} | mask lines/frame={int(mask[:, 0].sum())} "
        f"| actual_R≈{ny / max(mask[:, 0].sum(), 1):.3f}"
    )

    method_to_x: Dict[str, np.ndarray] = {}
    hist: List[Dict[str, float]] = []

    if "cg_sense_tikh" in cfg.methods_to_run:
        method_to_x["cg_sense_tikh"] = cg_sense_tikh(op, y, iters=cfg.cg_iters, lam=0.002)

    if "fista_sense_wavelet" in cfg.methods_to_run:
        method_to_x["fista_sense_wavelet"] = fista_wavelet(
            op,
            y,
            BaselineConfig(iters=cfg.fista_w_iters, step=0.8, lam=0.003),
        )

    if "fista_sense_tfft" in cfg.methods_to_run:
        method_to_x["fista_sense_tfft"] = fista_tfft(
            op,
            y,
            BaselineConfig(iters=cfg.fista_t_iters, step=0.8, lam=0.002),
        )

    if "pgd_sense_lowrank" in cfg.methods_to_run:
        method_to_x["pgd_sense_lowrank"] = pgd_lowrank(
            op, y, iters=cfg.lr_iters, step=0.15, lam0=0.02, decay=0.98
        )

    if "prop2_sense_pgd_v2" in cfg.methods_to_run:
        p2cfg = deepcopy(cfg.prop2)
        L = estimate_lipschitz(op, (full_kspace.shape[1], full_kspace.shape[2], full_kspace.shape[3]))
        p2cfg.tau0 = 0.10 / L
        print(f"  PROP2 Lipschitz L={L:.6e} -> tau0={p2cfg.tau0:.6e}")
        x_p2, hist = prop2_sense_pgd_improved(op, y, p2cfg)
        method_to_x["prop2_sense_pgd_v2"] = x_p2

    preds: Dict[str, np.ndarray] = {}
    rows: List[Dict[str, float]] = []

    for method, x in method_to_x.items():
        pred = center_crop(np.abs(x), cfg.crop_fraction)
        preds[method] = pred[len(pred) // 2]
        ms = [compute_metrics(ref[i], pred[i]) for i in range(ref.shape[0])]
        rows.append(
            {
                "file": file_stem,
                "R": int(R),
                "seed": int(seed),
                "method": method,
                "PSNR": float(np.mean([m["PSNR"] for m in ms])),
                "SSIM": float(np.mean([m["SSIM"] for m in ms])),
                "NRMSE": float(np.mean([m["NRMSE"] for m in ms])),
            }
        )

    fig_path = os.path.join(cfg.out_dir, f"{file_stem}_R{R}_seed{seed}_compare_all.png") if cfg.save_figures else ""
    show_this = cfg.display_figures and ((not cfg.display_only_seed0) or seed == 0)

    if preds:
        render_compare_figure(ref_frame, preds, f"{file_stem} | R={R} | seed={seed}", fig_path, show=show_this)

    return rows, preds, hist


def run_benchmark(
    h5_files: List[str],
    cfg: ExperimentConfig,
    run_cfg: RunConfig,
) -> Tuple[List[Dict[str, float]], Dict[str, List[Dict[str, float]]]]:
    os.makedirs(cfg.out_dir, exist_ok=True)
    all_rows: List[Dict[str, float]] = []
    debug_hist: Dict[str, List[Dict[str, float]]] = {}

    for h5_path in h5_files:
        file_stem = os.path.splitext(os.path.basename(h5_path))[0]
        print(f"\n=== Reading {h5_path} ===")
        full_kspace, param = read_ocmr_kspace(h5_path, run_cfg)
        print(f"Selected k-space shape [coil, frame, ky, kx]: {full_kspace.shape}")
        if param:
            print(f"Scan parameters: {param}")

        ky_center = int(param.get("ky_center", full_kspace.shape[2] // 2))
        sens = estimate_sens_maps_from_acs(full_kspace, acs_lines=cfg.acs_lines, ky_center=ky_center)

        if cfg.reference_mode.lower() == "rss":
            ref = make_reference_from_full_kspace(full_kspace, crop_fraction=cfg.crop_fraction)
        else:
            ref = make_reference_sense(full_kspace, sens, crop_fraction=cfg.crop_fraction)

        for R in cfg.accel_factors:
            print(f"\n--- Benchmarking {file_stem} | R={R} ---")
            for seed in cfg.seeds:
                rows, _, hist = run_single_case(full_kspace, ref, sens, file_stem, R, seed, cfg, ky_center)
                all_rows.extend(rows)
                debug_hist[f"{file_stem}_R{R}_seed{seed}"] = hist

    save_csv(os.path.join(cfg.out_dir, "per_run_metrics.csv"), all_rows)

    overall = sort_summary_rows(summarize_records(all_rows, group_keys=("method", "R")))
    by_file = sort_summary_rows(summarize_records(all_rows, group_keys=("file", "method", "R")))

    overall_pretty = add_pretty_columns(overall)
    by_file_pretty = add_pretty_columns(by_file)

    save_csv(os.path.join(cfg.out_dir, "summary_overall_numeric.csv"), overall)
    save_csv(os.path.join(cfg.out_dir, "summary_by_file_numeric.csv"), by_file)
    save_csv(os.path.join(cfg.out_dir, "summary_overall_pretty.csv"), overall_pretty)
    save_csv(os.path.join(cfg.out_dir, "summary_by_file_pretty.csv"), by_file_pretty)
    save_markdown_table(os.path.join(cfg.out_dir, "summary_overall_pretty.md"), overall_pretty)
    save_markdown_table(os.path.join(cfg.out_dir, "summary_by_file_pretty.md"), by_file_pretty)

    with open(os.path.join(cfg.out_dir, "iter_histories.json"), "w", encoding="utf-8") as f:
        json.dump(debug_hist, f, indent=2)

    return all_rows, {"overall": overall, "by_file": by_file}
