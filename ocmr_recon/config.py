from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass
class RunConfig:
    NUM_CASES: int = 2
    PREFER_FULLY_SAMPLED_PREFIX: str = "fs_"
    CASES_TO_RUN: Tuple[str, ...] = ()

    NUM_FRAMES: int = 10

    SEEDS: Tuple[int, ...] = (0, 1, 2)
    ACCEL_FACTORS: Tuple[int, ...] = (2, 4, 8, 12)
    ACS_LINES: int = 12

    NOISE_STD: float = 0.003
    CROP_FRACTION: float = 0.5

    DATA_DIR: str = "ocmr_data"
    OUT_DIR: str = "ocmr_results_R2_R4_R8_R12"

    SAVE_FIGURES: bool = False
    DISPLAY_FIGURES: bool = False
    DISPLAY_ONLY_SEED0: bool = True

    REFERENCE_MODE: str = "sense"

    # Keep only literature-backed baselines plus the proposed method.
    METHODS_TO_RUN: Tuple[str, ...] = (
        "cg_sense_tikh",
        "fista_sense_wavelet",
        "fista_sense_tfft",
        "pgd_sense_lowrank",
        "prop2_sense_pgd_v2",
    )

    CG_ITERS: int = 20
    FISTA_W_ITERS: int = 25
    FISTA_T_ITERS: int = 30
    LR_ITERS: int = 60
    PROP2_ITERS: int = 80

    SELECT_KZ: int = 0
    SELECT_SET: int = 0
    SELECT_SLICE: int = 0
    SELECT_REP: int = 0
    SELECT_AVG: int = 0

    TAU0: float = 0.10
    TAU_MIN: float = 1e-5
    BETA: float = 0.6
    ALPHA0: float = 0.20
    ALPHA1: float = 0.20
    DC_WEIGHT0: float = 0.70
    DC_WEIGHT1: float = 0.70
    LAM_W: float = 0.0015
    LAM_T: float = 0.0140
    LAM_H: float = 0.00015
    WAVELET: str = "db4"
    WAVELET_LEVEL: int = 2
    TEMPORAL_KEEP_LOW_BINS: int = 2
    HARTLEY_PROTECT_RADIUS: float = 0.25
    HARTLEY_GAMMA: float = 2.0
    WARM_START: str = "cg"
    EXACT_DC: bool = False


@dataclass
class Prop2Config:
    iters: int = 80
    tau0: float = 0.10
    tau_min: float = 1e-5
    beta: float = 0.6
    alpha0: float = 0.20
    alpha1: float = 0.20
    dc_weight0: float = 0.70
    dc_weight1: float = 0.70
    lam_w: float = 0.0015
    lam_t: float = 0.0140
    lam_h: float = 0.00015
    wavelet: str = "db4"
    wavelet_level: int = 2
    temporal_keep_low_bins: int = 2
    hartley_protect_radius: float = 0.25
    hartley_gamma: float = 2.0
    use_nesterov: bool = True
    use_restart: bool = True
    warm_start: str = "cg"
    monotone: bool = False
    continuation: bool = False
    outer_only_hartley: bool = True
    exact_dc: bool = False


@dataclass
class ExperimentConfig:
    accel_factors: Tuple[int, ...] = (2, 4, 8, 12)
    acs_lines: int = 12
    noise_std: float = 0.003
    seeds: Tuple[int, ...] = (0, 1, 2)
    crop_fraction: float = 0.5
    out_dir: str = "ocmr_results_R2_R4_R8_R12"
    save_figures: bool = False
    display_figures: bool = False
    display_only_seed0: bool = True
    reference_mode: str = "sense"

    methods_to_run: Tuple[str, ...] = (
        "cg_sense_tikh",
        "fista_sense_wavelet",
        "fista_sense_tfft",
        "pgd_sense_lowrank",
        "prop2_sense_pgd_v2",
    )

    cg_iters: int = 20
    fista_w_iters: int = 25
    fista_t_iters: int = 30
    lr_iters: int = 60

    prop2: Prop2Config = field(default_factory=Prop2Config)


CFG = RunConfig()


def build_experiment_config(cfg: RunConfig) -> ExperimentConfig:
    return ExperimentConfig(
        accel_factors=cfg.ACCEL_FACTORS,
        acs_lines=cfg.ACS_LINES,
        noise_std=cfg.NOISE_STD,
        seeds=cfg.SEEDS,
        crop_fraction=cfg.CROP_FRACTION,
        out_dir=cfg.OUT_DIR,
        save_figures=cfg.SAVE_FIGURES,
        display_figures=cfg.DISPLAY_FIGURES,
        display_only_seed0=cfg.DISPLAY_ONLY_SEED0,
        reference_mode=cfg.REFERENCE_MODE,
        methods_to_run=cfg.METHODS_TO_RUN,
        cg_iters=cfg.CG_ITERS,
        fista_w_iters=cfg.FISTA_W_ITERS,
        fista_t_iters=cfg.FISTA_T_ITERS,
        lr_iters=cfg.LR_ITERS,
        prop2=Prop2Config(
            iters=cfg.PROP2_ITERS,
            tau0=cfg.TAU0,
            tau_min=cfg.TAU_MIN,
            beta=cfg.BETA,
            alpha0=cfg.ALPHA0,
            alpha1=cfg.ALPHA1,
            dc_weight0=cfg.DC_WEIGHT0,
            dc_weight1=cfg.DC_WEIGHT1,
            lam_w=cfg.LAM_W,
            lam_t=cfg.LAM_T,
            lam_h=cfg.LAM_H,
            wavelet=cfg.WAVELET,
            wavelet_level=cfg.WAVELET_LEVEL,
            temporal_keep_low_bins=cfg.TEMPORAL_KEEP_LOW_BINS,
            hartley_protect_radius=cfg.HARTLEY_PROTECT_RADIUS,
            hartley_gamma=cfg.HARTLEY_GAMMA,
            use_nesterov=True,
            use_restart=True,
            warm_start=cfg.WARM_START,
            monotone=False,
            continuation=False,
            outer_only_hartley=True,
            exact_dc=cfg.EXACT_DC,
        ),
    )
