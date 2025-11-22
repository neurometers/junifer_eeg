"""General EEG referencing preprocessor."""

from typing import Any, ClassVar, Optional, Union

import mne
from junifer.api.decorators import register_preprocessor
from junifer.preprocess import BasePreprocessor
from mne.utils import logger


@register_preprocessor
class EEGReference(BasePreprocessor):
    """Apply EEG referencing (e.g., average reference).

    Parameters
    ----------
    ref_channels : str or list of str, optional
        Channels to use for reference.
        - 'average': Apply average reference (default).
        - 'REST': Apply Reference Electrode Standardization Technique.
        - list of str: List of channel names to use as reference.
        - None: Do not apply referencing (pass-through).
    projection : bool, optional
        If True, the reference is added as a projection and not applied immediately
        (unless apply_proj=True). If False, reference is applied immediately.
        Default: True (for average reference).
    apply_proj : bool, optional
        If True, apply projections after adding reference projection.
        Default: True.
    on : list of str, optional
        Data types to apply preprocessing to.
    """

    _DEPENDENCIES: ClassVar = {"mne"}

    def __init__(
        self,
        ref_channels: Union[str, list[str], None] = "average",
        projection: bool = True,
        apply_proj: bool = True,
        on: Optional[str] = None,
    ) -> None:
        """Initialize EEGReference."""
        self.ref_channels = ref_channels
        self.projection = projection
        self.apply_proj = apply_proj
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
        """Apply EEG referencing."""
        raw_or_epochs = input["data"]

        if not isinstance(raw_or_epochs, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise ValueError(
                "Input data must be mne.io.BaseRaw or mne.BaseEpochs"
            )

        # Make a copy to avoid modifying original
        # Note: set_eeg_reference modifies in-place or returns copy depending on copy param
        # We copy first to be safe and consistent with other preprocessors
        # However, set_eeg_reference usually returns the instance if copy=False
        # Let's copy first.
        inst = raw_or_epochs.copy()

        if self.ref_channels is not None:
            logger.info(f"Setting EEG reference to: {self.ref_channels}")

            # Handle average reference
            if self.ref_channels == "average":
                inst.set_eeg_reference(
                    ref_channels="average",
                    projection=self.projection,
                    verbose=False,
                )
            # Handle REST (if supported by MNE version, usually explicit function)
            elif self.ref_channels == "REST":
                inst.set_eeg_reference(
                    ref_channels="REST",
                    projection=self.projection,
                    verbose=False,
                )
            # Handle specific channels
            else:
                inst.set_eeg_reference(
                    ref_channels=self.ref_channels,
                    projection=self.projection,
                    verbose=False,
                )

            # Apply projections if requested
            if self.apply_proj:
                logger.info("Applying projections")
                inst.apply_proj()

        output = input.copy()
        output["data"] = inst

        return output, None
