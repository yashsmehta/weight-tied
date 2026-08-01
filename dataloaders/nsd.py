"""NSD (Natural Scenes Dataset) brain-data loading, NSD-only slice.

Ported from visreps/dataloaders/neural.py. Drops THINGS-behavior, TVSD,
NSD-synthetic, and Cusack-2025 support entirely (unused by this project),
and the legacy single-subject `load_nsd_data`/`get_neural_loader` (evals.py
only ever calls `load_all_nsd_data` directly).
"""
import logging
import os
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import utils

logger = logging.getLogger(__name__)

_NSD_REGION_MAP = {
    "early visual stream": "early",
    "ventral visual stream": "ventral",
    "V1": "V1",
    "V2": "V2",
    "V3": "V3",
    "hV4": "hV4",
    "FFA": "FFA",
    "PPA": "PPA",
}

_NSD_SUBJECTS = list(range(8))

# visreps hardcodes this path (single-machine deployment); made overridable
# here since this repo runs on a different machine.
_DEFAULT_NSD_STIMULI_HDF5_PATH = "/data/shared/datasets/allen2021.natural_scenes/nsddata_stimuli/stimuli/nsd/nsd_stimuli.hdf5"


def _nsd_stimuli_hdf5_path() -> str:
    return os.environ.get("NSD_STIMULI_HDF5_PATH", _DEFAULT_NSD_STIMULI_HDF5_PATH)


class _LazyHdf5Dict:
    """Dict-like wrapper around an HDF5 dataset that reads images on demand.

    Avoids loading all images into RAM. Compatible with _StimuliDataset,
    which accesses items via __getitem__.
    """

    def __init__(self, hdf5_path: str, dataset_name: str, indices):
        self._hdf5_path = hdf5_path
        self._dataset_name = dataset_name
        self._index_map = {str(idx): int(idx) for idx in indices}
        self._keys_sorted = sorted(self._index_map.keys(), key=lambda x: int(x))
        self._file = None

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self._hdf5_path, "r")
        return self._file

    def __contains__(self, key):
        return str(key) in self._index_map

    def __len__(self):
        return len(self._index_map)

    def keys(self):
        return self._keys_sorted

    def __getitem__(self, key):
        key_str = str(key)
        if key_str not in self._index_map:
            raise KeyError(key)
        return self._open()[self._dataset_name][self._index_map[key_str]]

    def __del__(self):
        if self._file is not None:
            self._file.close()


def load_all_nsd_data(cfg: Dict, subjects=None, regions=None) -> Dict:
    """Load NSD fMRI responses for requested subjects and regions.

    Args:
        subjects: List of subject indices to load (default: all 8).
        regions: List of full region names to load (default: both streams).

    Returns:
        dict with keys:
            - "regions": list of full region names loaded
            - "subjects": list of subject indices loaded
            - "neural": {region: {subj: {"train": {sid: resp}, "test": {sid: resp}}}}
            - "stimuli": {str(stim_id): np.ndarray} union of all stimulus images
            - "shared_test_ids": stimulus IDs shared across ALL subjects' test sets
    """
    subjects = subjects if subjects is not None else _NSD_SUBJECTS
    region_pairs = [(pkl_key, name) for name, pkl_key in _NSD_REGION_MAP.items()
                     if regions is None or name in regions]

    root = utils.get_env_var("NSD_DATA_DIR")
    nsd = utils.load_pickle(os.path.join(root, "nsd_data.pkl"))
    shared_ids = nsd["shared_ids"]

    neural = {}
    all_stimulus_ids = set()
    per_subject_test_ids = []

    for region_key, region_full in region_pairs:
        neural[region_full] = {}
        for subj in subjects:
            fmri_xr = nsd["data"][region_key][subj]
            stimulus_ids = [int(i) for i in fmri_xr.coords["stimulus"].values]
            all_stimulus_ids.update(stimulus_ids)

            train_ids = [str(i) for i in stimulus_ids if i not in shared_ids]
            test_ids = [str(i) for i in stimulus_ids if i in shared_ids]

            neural[region_full][subj] = {
                "train": {i: fmri_xr.sel(stimulus=int(i)).values for i in train_ids},
                "test": {i: fmri_xr.sel(stimulus=int(i)).values for i in test_ids},
            }

            if region_key == region_pairs[0][0]:
                per_subject_test_ids.append(set(test_ids))

    shared_test_ids = sorted(set.intersection(*per_subject_test_ids), key=int)

    # Lazy HDF5 wrapper -- reads images on demand, avoids loading ~70k images into RAM.
    stimuli = _LazyHdf5Dict(_nsd_stimuli_hdf5_path(), "imgBrick", all_stimulus_ids)

    region_names = [f for _, f in region_pairs]
    logger.info(
        f"Loaded NSD: {len(subjects)} subjects x {len(region_names)} regions, "
        f"{len(stimuli)} stimuli (lazy HDF5), {len(shared_test_ids)} shared test IDs"
    )

    return {
        "regions": region_names,
        "subjects": list(subjects),
        "neural": neural,
        "stimuli": stimuli,
        "shared_test_ids": shared_test_ids,
    }


class _StimuliDataset(Dataset):
    """PyTorch Dataset for stimuli, supporting file paths, ndarrays, or PIL images."""

    def __init__(self, stimuli, transform):
        self.keys = sorted(stimuli.keys())
        self.stimuli = stimuli  # reference, not a copy -- may be a lazy dict
        self.tr = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.keys)

    def _load_and_transform(self, data_or_path: Any, key: str):
        if isinstance(data_or_path, str):
            img = Image.open(data_or_path).convert("RGB")
        elif isinstance(data_or_path, np.ndarray):
            img = Image.fromarray(data_or_path.astype("uint8"), "RGB")
        elif isinstance(data_or_path, Image.Image):
            img = data_or_path.convert("RGB") if data_or_path.mode != "RGB" else data_or_path
        else:
            raise TypeError(f"Unsupported data type {type(data_or_path)} for key {key}")
        return self.tr(img)

    def __getitem__(self, idx):
        key = self.keys[idx]
        transformed_img = self._load_and_transform(self.stimuli[key], key)
        return transformed_img, key


def custom_collate_fn(batch: List[Tuple[torch.Tensor, str]]) -> Tuple[torch.Tensor, List[str]]:
    imgs, keys = zip(*batch)
    return torch.stack(imgs), list(keys)


def _make_loader(stimuli, transform, batch, workers):
    return DataLoader(
        _StimuliDataset(stimuli, transform),
        batch_size=batch,
        shuffle=False,
        num_workers=workers,
        collate_fn=custom_collate_fn,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        prefetch_factor=2 if workers > 0 else None,
    )
