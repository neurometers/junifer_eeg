"""ICM Local-Global Data Grabber for junifer_eeg."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataGrabber


@register_datagrabber
class ICMLGDataGrabber(PatternDataGrabber):
    """Data grabber for ICM Local-Global paradigm with trigger processing.

    This data grabber extends PatternDataGrabber to automatically apply
    ICM LG specific processing (trigger processing, channel filtering)
    to all EEG file types during data loading.
    """

    def __init__(
        self,
        datadir: Union[str, Path],
        types: Optional[List[str]] = None,
        patterns: Optional[Dict] = None,
        replacements: Optional[List[str]] = None,
        equipment_type: str = "egi",
        **kwargs,
    ):
        """Initialize ICM LG Data Grabber.

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
        equipment_type : str, optional
            Equipment type for trigger processing. Default: "egi".
            Options: "egi", "brainvision", "ant", "biosemi", "edf".
        **kwargs
            Additional arguments passed to PatternDataGrabber.
        """
        # Set defaults for ICM LG paradigm
        if types is None:
            types = ["EEG"]

        if patterns is None:
            patterns = {
                "EEG": {"pattern": "{subject}/*.mff", "space": "native"}
            }

        if replacements is None:
            replacements = ["subject"]

        # Store equipment type for datareader
        self.equipment_type = equipment_type

        super().__init__(
            datadir=datadir,
            types=types,
            patterns=patterns,
            replacements=replacements,
            **kwargs,
        )

    def __getitem__(self, element: Union[str, Dict]) -> Dict:
        """Get data for a specific element with ICM LG trigger processing.

        Parameters
        ----------
        element : str or dict
            Element identifier or element dictionary.

        Returns
        -------
        dict
            Dictionary containing processed EEG data with ICM LG triggers.
        """
        # Get data using parent class (reads raw file)
        data = super().__getitem__(element)

        # Apply ICM LG trigger processing using ICMLGDataReader
        # Note: When a datagrabber is present, the datareader in YAML is not used,
        # so we process triggers here
        if "EEG" in data:
            from ..datareader.icm_lg_datareader import ICMLGDataReader

            icm_datareader = ICMLGDataReader(
                equipment_type=self.equipment_type,
                apply_montage=True,
                process_triggers=True,
            )
            processed_data = icm_datareader._fit_transform(data)
            data.update(processed_data)

            # Debug: Verify annotations are in the returned data
            from collections import Counter

            from junifer.utils import logger

            raw_processed = data["EEG"]["data"]
            desc_counts = Counter(raw_processed.annotations.description)
            icm_conds = {
                d: c
                for d, c in desc_counts.items()
                if d in {"HSTD", "HDVT", "LSGS", "LSGD", "LDGD", "LDGS"}
            }
            logger.info(
                f"[DATAGRABBER] Returning data with {sum(icm_conds.values())} ICM events"
            )

            # CRITICAL: Remove 'path' key to prevent Junifer from reloading the file
            # and losing the processed annotations. The data is already loaded in memory.
            if "path" in data["EEG"]:
                del data["EEG"]["path"]

        return data
