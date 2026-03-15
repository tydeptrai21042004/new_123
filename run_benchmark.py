from ocmr_recon.config import CFG, build_experiment_config
from ocmr_recon.data import ensure_support_files, pick_cases, download_case_by_name
from ocmr_recon.experiment import run_benchmark
from ocmr_recon.reporting import display_summary_tables, manuscript_ready_summary
import os
import time


def main() -> None:
    t0 = time.time()
    os.makedirs(CFG.DATA_DIR, exist_ok=True)
    os.makedirs(CFG.OUT_DIR, exist_ok=True)

    attrs_path = ensure_support_files(CFG.DATA_DIR)
    case_names = pick_cases(attrs_path, CFG)

    print("\nSelected cases:")
    for c in case_names:
        print(" -", c)

    local_files = [download_case_by_name(name, CFG.DATA_DIR) for name in case_names]
    exp_cfg = build_experiment_config(CFG)

    _, summaries = run_benchmark(local_files, exp_cfg, CFG)

    print("\nSaved results to:", CFG.OUT_DIR)
    print("\n=== Overall summary (mean ± std across all files and seeds) ===")
    print(manuscript_ready_summary(summaries["overall"]))

    display_summary_tables(CFG.OUT_DIR)
    print(f"\nTotal elapsed: {(time.time() - t0) / 60.0:.2f} min")


if __name__ == "__main__":
    main()
