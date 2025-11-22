"""General EEG interpolation preprocessor."""

from typing import Any, ClassVar, Optional

import mne
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
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    _DEPENDENCIES: ClassVar = {"mne"}

    def __init__(
        self,
        method: Optional[dict] = None,
        reset_bads: bool = True,
        origin: str = "auto",
        on: Optional[str] = None,
    ) -> None:
        """Initialize EEGInterpolation."""
        self.method = method or {"eeg": "spline"}
        self.reset_bads = reset_bads
        self.origin = origin
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

        return output, None
