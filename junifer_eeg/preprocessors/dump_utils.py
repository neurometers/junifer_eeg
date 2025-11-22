"""Utilities for dumping preprocessing data using junifer's DataObjectDumper."""

from pathlib import Path
from typing import Any, Dict

import mne
from junifer.pipeline._data_object_dumper import DataObjectDumper
from mne.utils import logger


def dump_preprocessing_data(
    data: Dict[str, Any],
    dump_path: str,
) -> None:
    """Dump preprocessing data using junifer's DataObjectDumper.

    This function uses junifer's DataObjectDumper which automatically
    uses the registered MNE asset dumpers (MNEEpochsAsset, MNERawAsset)
    based on the data type.

    Parameters
    ----------
    data : dict
        Data dictionary containing 'data' key with MNE object and 'meta' key
        with metadata. All metadata will be stored in the data.yaml file.
    dump_path : str
        Full path (including filename) for dumping. Can contain {element} placeholder
        which will be replaced with the actual element from data['meta']['element'].
        The filename (without extension) will be used as the stage name for DataObjectDumper.
    """
    if not dump_path:
        return

    # Get element from data if available
    element = data.get("meta", {}).get("element", "unknown_element")
    if isinstance(element, dict):
        element = "unknown_element"

    # Replace {element} placeholder if present
    if "{element}" in dump_path:
        dump_path = dump_path.format(element=element)

    dump_path_obj = Path(dump_path)
    # Ensure parent directories exist
    dump_path_obj.parent.mkdir(parents=True, exist_ok=True)

    # Extract stage name from filename (without extension)
    # Try to extract a meaningful stage name by removing common prefixes/suffixes
    # e.g., "01_equipment_filtered_eeg" -> "equipment_filtered"
    filename_stem = dump_path_obj.stem
    # Remove leading numbers and underscores (e.g., "01_" -> "")
    stage_name = filename_stem
    parts = filename_stem.split("_")
    if parts and parts[0].isdigit():
        # Remove numeric prefix
        stage_name = "_".join(parts[1:])
    # Remove trailing "_eeg" if present
    if stage_name.endswith("_eeg"):
        stage_name = stage_name[:-4]
    # If nothing left, use the original stem
    if not stage_name:
        stage_name = filename_stem

    # Use junifer's DataObjectDumper for MNE objects
    # This automatically uses the registered MNEEpochsAsset or MNERawAsset
    # DataObjectDumper expects data dict with structure: {"EEG": {"data": <mne_obj>, "path": <path>, "meta": {...}, ...}}
    # All metadata will be automatically stored in the data.yaml file created by DataObjectDumper
    if isinstance(data.get("data"), (mne.io.BaseRaw, mne.BaseEpochs)):
        dumper = DataObjectDumper()
        # Structure data for DataObjectDumper: it expects top-level keys like "EEG"
        # with nested dict containing "data", "path", "meta", and other metadata
        eeg_data_dict = {
            "EEG": {
                "data": data["data"],
                "path": str(dump_path_obj),
                **{k: v for k, v in data.items() if k != "data"},
            }
        }
        # Use the full path as base_path, DataObjectDumper will create subdirectory with stage_name
        # and automatically save all metadata (including meta) to data.yaml
        dumper.dump(eeg_data_dict, dump_path_obj, stage_name)
        logger.info(f"Dumped EEG data using DataObjectDumper: {dump_path_obj}")
