"""Simple EEG data grabber following junifer documentation pattern."""

from pathlib import Path

from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataGrabber


@register_datagrabber
class EEGDataGrabber(PatternDataGrabber):
    """Simple EEG data grabber using MNE.

    Follows the junifer extension documentation pattern for DataGrabbers.
    Uses BOLD data type as it's the closest to time-series EEG data.
    """

    def __init__(self, datadir: str | Path) -> None:
        """Initialize the EEGDataGrabber.

        Parameters
        ----------
        datadir : str or Path
            Path to the directory containing EEG files.

        """
        types = ["BOLD"]
        patterns = {
            "BOLD": {
                "pattern": "{subject}_eeg.edf",
                "space": "native",
            },
        }
        replacements = ["subject"]

        super().__init__(
            datadir=datadir,
            types=types,
            patterns=patterns,
            replacements=replacements,
        )
