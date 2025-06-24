"""EEG data loader preprocessor."""

from pathlib import Path
from typing import Any, ClassVar

import mne
from junifer.api.decorators import register_preprocessor
from junifer.preprocess import BasePreprocessor


@register_preprocessor
class EEGLoader(BasePreprocessor):
    """Load EEG data from file paths into MNE Raw objects.

    This preprocessor bridges the gap between DataGrabbers
    (which provide file paths) and Markers (which expect loaded
    data objects). Uses BOLD data type as it's the closest to
    time-series EEG data.
    """

    _DEPENDENCIES: ClassVar = {"mne"}

    def __init__(self, on: str | None = None) -> None:
        """Initialize the EEGLoader preprocessor."""
        super().__init__(on=on)

    def get_valid_inputs(self) -> list[str]:
        """Get valid data types for input."""
        return ["BOLD"]

    def get_output_type(self, input_type: str) -> str:
        """Get output data type."""
        return input_type

    def preprocess(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Load EEG data from file path.

        Parameters
        ----------
        input : dict
            Input data with 'path' key pointing to EDF file.
        extra_input : dict, optional
            Additional input data.

        Returns
        -------
        tuple
            Modified input with 'raw_object' key and None for extra output.

        """
        file_path = Path(input["path"])

        # Load EEG data using MNE
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)

        # Add the raw object to the input
        input["raw_object"] = raw

        return input, None
