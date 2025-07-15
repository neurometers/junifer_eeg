"""Simple EEG filter preprocessor."""

from typing import Any, ClassVar

from junifer.api.decorators import register_preprocessor
from junifer.preprocess import BasePreprocessor


@register_preprocessor
class EEGFilter(BasePreprocessor):
    """Simple EEG filter using MNE."""

    _DEPENDENCIES: ClassVar = {"mne"}

    def __init__(
        self,
        low_freq: float | None = None,
        high_freq: float | None = None,
        on: str | None = None,
    ) -> None:
        """Initialize EEGFilter."""
        self.low_freq = low_freq
        self.high_freq = high_freq
        super().__init__(on=on)

    def get_valid_inputs(self) -> list[str]:
        """Get valid data types."""
        return ["EEG"]  # Now we can use EEG as the data type

    def get_output_type(self, input_type: str) -> str:
        """Get output type."""
        return input_type

    def preprocess(
        self,
        input: dict[str, Any],
        extra_input: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any] | None]:
        """Filter EEG data."""
        raw = input["data"]

        if self.low_freq is not None or self.high_freq is not None:
            raw = raw.filter(
                l_freq=self.low_freq,
                h_freq=self.high_freq,
                verbose=False,
            )

        input["data"] = raw
        return input, None
