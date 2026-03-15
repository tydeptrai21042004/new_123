# OCMR dynamic MRI benchmark (literature-supported baselines only)

This project refactors your original single-file script into a small package and keeps only:

- `R = {2, 4, 8, 12}` by default
- literature-supported baseline families for undersampled MRI / dynamic MRI
- your proposed Hartley-inspired method

## Baselines kept

- `cg_sense_tikh`
- `fista_sense_wavelet`
- `fista_sense_tfft`
- `pgd_sense_lowrank`
- `prop2_sense_pgd_v2`

## Baselines removed

The following methods were removed because they were only **paper-inspired surrogates** in the previous version, not clean literature-backed baselines in their current implementation:

- `ista_sense_wavelet_exactdc`
- `pgd_sense_ls_temporal`
- `pgd_sense_ktslr_like`

## Literature support used for the kept baselines

- **SENSE / CG-SENSE + Tikhonov:** Pruessmann et al., 1999
- **Wavelet sparse MRI baseline:** Lustig et al., 2007
- **Temporal Fourier sparsity baseline for dynamic MRI:** Lustig et al. (k-t SPARSE, 2006) and Jung et al. (k-t FOCUSS, 2009)
- **Low-rank dynamic MRI baseline:** Lingala et al., 2011 and Otazo et al., 2015

See `BASELINE_PAPERS.md` for a short mapping from code method name to literature.

## Directory structure

```text
ocmr_recon_project/
├── BASELINE_PAPERS.md
├── README.md
├── requirements.txt
├── run_benchmark.py
└── ocmr_recon/
    ├── __init__.py
    ├── config.py
    ├── utils.py
    ├── data.py
    ├── ops.py
    ├── prox.py
    ├── reporting.py
    ├── experiment.py
    └── methods/
        ├── __init__.py
        ├── baselines.py
        └── proposed.py
```

## Install

```bash
python -m pip install -r requirements.txt
```

## Run

```bash
python run_benchmark.py
```

## Quick edits

Open `ocmr_recon/config.py` and change:

- `ACCEL_FACTORS`
- `METHODS_TO_RUN`
- `NUM_CASES`
- `CASES_TO_RUN`
- `NUM_FRAMES`
- output paths

## Output files

The script writes:

- `per_run_metrics.csv`
- `summary_overall_numeric.csv`
- `summary_by_file_numeric.csv`
- `summary_overall_pretty.csv`
- `summary_by_file_pretty.csv`
- `summary_overall_pretty.md`
- `summary_by_file_pretty.md`
- `iter_histories.json`
