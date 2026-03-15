# Baseline-to-literature mapping

All remaining baselines below are kept because they correspond to published reconstruction families in undersampled MRI / dynamic MRI. The proposed method is listed separately.

## 1) `cg_sense_tikh`
Literature family: SENSE / parallel imaging with quadratic regularization.
Primary source:
- Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: sensitivity encoding for fast MRI. Magn Reson Med. 1999.

## 2) `fista_sense_wavelet`
Literature family: compressed sensing MRI with spatial wavelet sparsity.
Primary source:
- Lustig M, Donoho D, Pauly JM. Sparse MRI: The application of compressed sensing for rapid MR imaging. Magn Reson Med. 2007.

## 3) `fista_sense_tfft`
Literature family: dynamic MRI with temporal Fourier sparsity.
Primary sources:
- Lustig M, Santos J, Donoho D, Pauly JM. k-t SPARSE: High frame rate dynamic MRI exploiting spatio-temporal sparsity. ISMRM 2006.
- Jung H, Sung K, Nayak KS, Kim EY, Ye JC. k-t FOCUSS: A general compressed sensing framework for high resolution dynamic MRI. Magn Reson Med. 2009.

## 4) `pgd_sense_lowrank`
Literature family: low-rank dynamic MRI.
Primary sources:
- Lingala SG, Hu Y, DiBella E, Jacob M. Accelerated dynamic MRI exploiting sparsity and low-rank structure: k-t SLR. IEEE Trans Med Imaging. 2011.
- Otazo R, Candès E, Sodickson DK. Low-rank plus sparse matrix decomposition for accelerated dynamic MRI with separation of background and dynamic components. Magn Reson Med. 2015.

## 5) `prop2_sense_pgd_v2`
Your proposed method. This is not claimed to be a baseline from prior literature.
