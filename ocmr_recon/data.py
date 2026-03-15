from __future__ import annotations

import csv
import math
import os
import urllib.request
from typing import Dict, List, Optional, Tuple

import ismrmrd
import ismrmrd.xsd
import numpy as np

from .config import RunConfig

DEFAULT_ATTRS_URL = "https://raw.githubusercontent.com/MRIOSU/OCMR/master/ocmr_data_attributes.csv"
DEFAULT_CASE_BASE_URL = "https://ocmr.s3.us-east-2.amazonaws.com/data/"


def _download_progress(blocks: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        done_mb = blocks * block_size / (1024 * 1024)
        print(f"\rDownloaded {done_mb:.1f} MB", end="", flush=True)
        return
    done = min(blocks * block_size, total_size)
    pct = 100.0 * done / total_size
    print(
        f"\r{pct:6.2f}% | {done / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB",
        end="",
        flush=True,
    )


def download_file(url: str, dst_path: str, overwrite: bool = False) -> str:
    os.makedirs(os.path.dirname(dst_path) or ".", exist_ok=True)
    if os.path.exists(dst_path) and not overwrite:
        print(f"Already exists: {dst_path}")
        return dst_path
    print(f"Downloading {url} -> {dst_path}")
    urllib.request.urlretrieve(url, dst_path, reporthook=_download_progress)
    print()
    return dst_path


def ensure_support_files(data_dir: str) -> str:
    os.makedirs(data_dir, exist_ok=True)
    attrs_path = os.path.join(data_dir, "ocmr_data_attributes.csv")
    download_file(DEFAULT_ATTRS_URL, attrs_path, overwrite=False)
    return attrs_path


def download_case_by_name(case_name: str, data_dir: str) -> str:
    dst = os.path.join(data_dir, case_name)
    return download_file(DEFAULT_CASE_BASE_URL + case_name, dst, overwrite=False)


def find_h5_name_in_row(row: Dict[str, str]) -> Optional[str]:
    for _, val in row.items():
        s = str(val).strip()
        if s.lower().endswith(".h5"):
            return s
    return None


def load_case_names_from_attrs(attrs_path: str) -> List[str]:
    out: List[str] = []
    with open(attrs_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = find_h5_name_in_row(row)
            if name:
                out.append(name)

    seen = set()
    uniq = []
    for x in out:
        if x not in seen:
            seen.add(x)
            uniq.append(x)
    return uniq


def pick_cases(attrs_path: str, cfg: RunConfig) -> List[str]:
    if cfg.CASES_TO_RUN:
        return list(cfg.CASES_TO_RUN)

    names = load_case_names_from_attrs(attrs_path)
    preferred = [x for x in names if os.path.basename(x).lower().startswith(cfg.PREFER_FULLY_SAMPLED_PREFIX.lower())]
    chosen = preferred[: cfg.NUM_CASES] if len(preferred) >= cfg.NUM_CASES else names[: cfg.NUM_CASES]

    if not chosen:
        raise RuntimeError("Could not auto-select any OCMR cases from the attributes CSV.")
    return chosen


def read_ocmr_ismrmrd(filename: str) -> Tuple[np.ndarray, Dict[str, object]]:
    if not os.path.isfile(filename):
        raise FileNotFoundError(filename)

    dset = ismrmrd.Dataset(filename, "dataset", create_if_needed=False)
    header = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())
    enc = header.encoding[0]

    eNx = enc.encodedSpace.matrixSize.x
    eNz = enc.encodedSpace.matrixSize.z
    eNy = enc.encodingLimits.kspace_encoding_step_1.maximum + 1

    eFOVx = enc.encodedSpace.fieldOfView_mm.x
    eFOVy = enc.encodedSpace.fieldOfView_mm.y
    eFOVz = enc.encodedSpace.fieldOfView_mm.z

    param: Dict[str, object] = {}
    try:
        param["TRes"] = str(header.sequenceParameters.TR)
        param["TE"] = str(header.sequenceParameters.TE)
        param["TI"] = str(header.sequenceParameters.TI)
        param["echo_spacing"] = str(header.sequenceParameters.echo_spacing)
        param["flipAngle_deg"] = str(header.sequenceParameters.flipAngle_deg)
        param["sequence_type"] = header.sequenceParameters.sequence_type
    except Exception:
        pass
    param["FOV"] = [eFOVx, eFOVy, eFOVz]
    param["kspace_dim"] = ("kx", "ky", "kz", "coil", "phase", "set", "slice", "rep", "avg")

    try:
        ky_center = int(enc.encodingLimits.kspace_encoding_step_1.center)
    except Exception:
        ky_center = eNy // 2
    param["ky_center"] = ky_center

    def _get_lim(attr_name: str) -> int:
        try:
            lim = getattr(enc.encodingLimits, attr_name)
            return int(lim.maximum) + 1
        except Exception:
            return 1

    nSlices = _get_lim("slice")
    nReps = _get_lim("repetition")
    nPhases = _get_lim("phase")
    nSets = _get_lim("set")
    nAverage = _get_lim("average")
    nCoils = header.acquisitionSystemInformation.receiverChannels

    firstacq = 0
    for acqnum in range(dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        if acq.isFlagSet(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            continue
        firstacq = acqnum
        print("Imaging acquisition starts at acq", acqnum)
        break

    kx_prezp = 0
    acq_first = dset.read_acquisition(firstacq)
    if acq_first.center_sample * 2 < eNx:
        kx_prezp = eNx - acq_first.number_of_samples

    all_data = np.zeros((eNx, eNy, eNz, nCoils, nPhases, nSets, nSlices, nReps, nAverage), dtype=np.complex64)

    pilottone = 0
    try:
        upl = header.userParameters.userParameterLong
        for item in upl:
            if item.name == "PilotTone":
                pilottone = int(item.value)
                break
    except Exception:
        pilottone = 0

    if pilottone == 1:
        print("Pilot Tone is on, discarding the first 3 and last 1 k-space point for each line")

    for acqnum in range(firstacq, dset.number_of_acquisitions()):
        acq = dset.read_acquisition(acqnum)
        if pilottone == 1 and acq.data.shape[1] >= 4:
            acq.data[:, [0, 1, 2, acq.data.shape[1] - 1]] = 0

        y = int(acq.idx.kspace_encode_step_1)
        z = int(acq.idx.kspace_encode_step_2)
        phase = int(acq.idx.phase)
        set_idx = int(acq.idx.set)
        slice_idx = int(acq.idx.slice)
        rep = int(acq.idx.repetition)
        avg = int(acq.idx.average)

        all_data[kx_prezp:, y, z, :, phase, set_idx, slice_idx, rep, avg] = np.transpose(acq.data)

    return all_data, param


def select_cine_block(k9: np.ndarray, cfg: RunConfig) -> np.ndarray:
    _, _, kz, _, _, nset, nslice, nrep, navg = k9.shape

    iz = min(cfg.SELECT_KZ, max(0, kz - 1))
    iset = min(cfg.SELECT_SET, max(0, nset - 1))
    islice = min(cfg.SELECT_SLICE, max(0, nslice - 1))
    irep = min(cfg.SELECT_REP, max(0, nrep - 1))
    iavg = min(cfg.SELECT_AVG, max(0, navg - 1))

    block = k9[:, :, iz, :, :, iset, islice, irep, iavg]
    block = np.transpose(block, (2, 3, 1, 0)).astype(np.complex64)  # [coil, phase, ky, kx]

    if cfg.NUM_FRAMES > 0 and block.shape[1] > cfg.NUM_FRAMES:
        block = block[:, : cfg.NUM_FRAMES, :, :]

    return block


def read_ocmr_kspace(filename: str, cfg: RunConfig) -> Tuple[np.ndarray, Dict[str, object]]:
    k9, param = read_ocmr_ismrmrd(filename)
    return select_cine_block(k9, cfg), param
