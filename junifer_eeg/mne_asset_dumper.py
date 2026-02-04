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

    # Register dumpers for all MNE epochs types
    AssetDumperDispatcher()[mne.Epochs] = MNEEpochsAsset
    AssetDumperDispatcher()[mne.epochs.EpochsFIF] = MNEEpochsAsset

    # Register base raw type
    AssetDumperDispatcher()[mne.io.BaseRaw] = MNERawAsset

    # Register all specific raw format types
    # These need explicit registration as some don't match BaseRaw properly
    try:
        import mne.io.edf.edf

        AssetDumperDispatcher()[mne.io.edf.edf.RawEDF] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.eeglab.eeglab

        AssetDumperDispatcher()[mne.io.eeglab.eeglab.RawEEGLAB] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.brainvision.brainvision

        AssetDumperDispatcher()[
            mne.io.brainvision.brainvision.RawBrainVision
        ] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.fif.raw

        AssetDumperDispatcher()[mne.io.fif.raw.Raw] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.egi.egimff

        AssetDumperDispatcher()[mne.io.egi.egimff.RawMff] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.cnt.cnt

        AssetDumperDispatcher()[mne.io.cnt.cnt.RawCNT] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.gdf.gdf

        AssetDumperDispatcher()[mne.io.gdf.gdf.RawGDF] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.nihon.nihon

        AssetDumperDispatcher()[mne.io.nihon.nihon.RawNihon] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.kit.kit

        AssetDumperDispatcher()[mne.io.kit.kit.RawKIT] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.curry.curry

        AssetDumperDispatcher()[mne.io.curry.curry.RawCurry] = MNERawAsset
    except (ImportError, AttributeError):
        pass

    try:
        import mne.io.nicolet.nicolet

        AssetDumperDispatcher()[mne.io.nicolet.nicolet.RawNicolet] = (
            MNERawAsset
        )
    except (ImportError, AttributeError):
        pass
