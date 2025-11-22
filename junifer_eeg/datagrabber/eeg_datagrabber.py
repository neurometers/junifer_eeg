"""Vanilla EEG Data Grabber for junifer_eeg.

This is a simple data grabber that uses EEGDataReader to handle
common functionality needed for markers (equipment detection, montage setting).
Assumes data has already been processed and is ready as simple MNE objects.
"""

from pathlib import Path
from typing import Dict, List, Optional, Union

from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataGrabber


@register_datagrabber
class EEGDataGrabber(PatternDataGrabber):
    """Vanilla EEG Data Grabber.

    Simple data grabber that applies common processing needed for markers:
    - Equipment detection
    - Montage setting

    Assumes data is already processed and ready as MNE objects.
    """

    def __init__(
        self,
        datadir: Union[str, Path],
        types: Optional[List[str]] = None,
        patterns: Optional[Dict] = None,
        replacements: Optional[List[str]] = None,
        **kwargs,
    ):
        """Initialize EEG Data Grabber.

        Parameters
        ----------
        datadir : str or Path
            Path to the data directory.
        types : list of str, optional
            Data types to grab. If None, defaults to ["EEG"].
        patterns : dict, optional
            Patterns for each data type. If None, uses default EEG patterns.
        replacements : list of str, optional
            Replacement variables. If None, defaults to ["subject"].
        **kwargs
            Additional arguments passed to PatternDataGrabber.
        """
        # Set defaults
        if types is None:
            types = ["EEG"]

        if patterns is None:
            patterns = {
                "EEG": {"pattern": "{subject}/*.fif", "space": "native"}
            }

        if replacements is None:
            replacements = ["subject"]

        super().__init__(
            datadir=datadir,
            types=types,
            patterns=patterns,
            replacements=replacements,
            **kwargs,
        )

    def __getitem__(self, element: Union[str, Dict]) -> Dict:
        """Get data for a specific element with equipment detection.

        Parameters
        ----------
        element : str or dict
            Element identifier or element dictionary.

        Returns
        -------
        dict
            Dictionary containing EEG data with equipment information.
        """
        # Get data using parent class
        data = super().__getitem__(element)

        # Apply common processing using EEGDataReader
        if "EEG" in data:
            from ..datareader.eeg_datareader import EEGDataReader

            eeg_datareader = EEGDataReader()
            processed_data = eeg_datareader._fit_transform(data)
            data.update(processed_data)

        return data
