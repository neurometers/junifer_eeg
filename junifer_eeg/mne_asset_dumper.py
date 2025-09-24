"""MNE asset dumper for junifer DataObjectDumper."""

from pathlib import Path

import mne
from junifer.pipeline._data_object_dumper import BaseDataDumpAsset


class MNEEpochsAsset(BaseDataDumpAsset):
    """Class for MNE Epochs dumper."""

    def dump(self) -> None:
        """Dump MNE Epochs to FIF format."""
        fif_path = (
            self.path_without_ext.parent
            / f"{self.path_without_ext.name}-epo.fif"
        )
        self.data.save(fif_path, overwrite=True)

    @classmethod
    def load(cls: "MNEEpochsAsset", path: Path) -> mne.Epochs:
        """Load MNE Epochs from FIF format."""
        return mne.read_epochs(path, verbose=False)


class MNERawAsset(BaseDataDumpAsset):
    """Class for MNE Raw dumper."""

    def dump(self) -> None:
        """Dump MNE Raw to FIF format."""
        fif_path = (
            self.path_without_ext.parent
            / f"{self.path_without_ext.name}_raw.fif"
        )
        self.data.save(fif_path, overwrite=True)

    @classmethod
    def load(cls: "MNERawAsset", path: Path) -> mne.io.BaseRaw:
        """Load MNE Raw from FIF format."""
        return mne.io.read_raw_fif(path, verbose=False)


def register_mne_dumpers():
    """Register MNE data types with junifer's DataObjectDumper."""
    import mne
    from junifer.pipeline._data_object_dumper import (
        AssetDumperDispatcher,
    )

    # Register dumpers for all MNE raw data types
    AssetDumperDispatcher()[mne.Epochs] = MNEEpochsAsset
    AssetDumperDispatcher()[mne.epochs.EpochsFIF] = (
        MNEEpochsAsset  # Add EpochsFIF specifically
    )
    AssetDumperDispatcher()[mne.io.BaseRaw] = MNERawAsset

    # Register MFF format specifically
    import mne.io.egi.egimff

    AssetDumperDispatcher()[mne.io.egi.egimff.RawMff] = MNERawAsset
