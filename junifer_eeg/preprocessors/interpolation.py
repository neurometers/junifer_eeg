"""General EEG interpolation preprocessor."""

from typing import Any, ClassVar, Optional

import mne
import numpy as np
from junifer.api.decorators import register_preprocessor
from junifer.preprocess import BasePreprocessor
from mne.utils import logger


@register_preprocessor
class EEGInterpolation(BasePreprocessor):
    """Interpolate bad channels.

    Parameters
    ----------
    method : dict, optional
        Interpolation method. See mne.io.Raw.interpolate_bads documentation.
        Default: {'eeg': 'spline'}.
    reset_bads : bool, optional
        If True, clear the list of bad channels after interpolation.
        Default: True.
    origin : array-like, shape (3,) or str, optional
        Origin of the sphere. See mne.io.Raw.interpolate_bads documentation.
        Default: 'auto'.
    dump_path : str, optional
        Full path (including filename) for dumping. If None, no dumping.
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    _DEPENDENCIES: ClassVar = {"mne"}

    def __init__(
        self,
        method: Optional[dict] = None,
        reset_bads: bool = True,
        origin: str = "auto",
        dump_path: Optional[str] = None,
        on: Optional[str] = None,
    ) -> None:
        """Initialize EEGInterpolation."""
        self.method = method or {"eeg": "spline"}
        self.reset_bads = reset_bads
        self.origin = origin
        self.dump_path = dump_path
        super().__init__(on=on)

    def get_valid_inputs(self) -> list[str]:
        """Get valid data types."""
        return ["EEG"]

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Interpolate bad channels."""
        raw_or_epochs = input["data"]

        if not isinstance(raw_or_epochs, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise ValueError(
                "Input data must be mne.io.BaseRaw or mne.BaseEpochs"
            )

        # Make a copy to avoid modifying original
        inst = raw_or_epochs.copy()

        # First, find and drop ALL EEG channels with invalid positions
        # (e.g., "Vertex Reference") - these break MNE's interpolation matrix
        eeg_chs_to_drop = []
        for i, ch_name in enumerate(inst.ch_names):
            ch_info = inst.info["chs"][i]
            # Only check EEG channels (kind == 2)
            if ch_info.get("kind") == 2:  # FIFFV_EEG_CH
                pos = ch_info.get("loc", None)
                if pos is not None and len(pos) >= 3:
                    pos_3d = pos[:3]
                    if np.any(np.isnan(pos_3d)) or np.any(np.isinf(pos_3d)):
                        eeg_chs_to_drop.append(ch_name)
                    elif np.allclose(pos_3d, 0):
                        eeg_chs_to_drop.append(ch_name)

        if eeg_chs_to_drop:
            logger.warning(
                f"Dropping {len(eeg_chs_to_drop)} EEG channels without valid "
                f"positions: {eeg_chs_to_drop}"
            )
            # Remove from bads list first if present
            inst.info["bads"] = [
                b for b in inst.info["bads"] if b not in eeg_chs_to_drop
            ]
            inst.drop_channels(eeg_chs_to_drop)

        if inst.info["bads"]:
            logger.info(
                f"Interpolating {len(inst.info['bads'])} bad channels: {inst.info['bads']}"
            )
            inst.interpolate_bads(
                reset_bads=self.reset_bads,
                method=self.method,
                origin=self.origin,
                verbose=False,
            )
        else:
            logger.info("No bad channels to interpolate.")

        output = input.copy()
        output["data"] = inst

        # Dump data if requested
        if self.dump_path:
            from .dump_utils import dump_preprocessing_data

            dump_preprocessing_data(output, self.dump_path)

        return output, None
