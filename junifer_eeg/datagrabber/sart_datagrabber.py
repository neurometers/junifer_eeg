"""SART Data Grabber for junifer_eeg."""

from pathlib import Path
from typing import Dict, List, Optional, Union

from junifer.api.decorators import register_datagrabber
from junifer.datagrabber import PatternDataGrabber


@register_datagrabber
class SARTDataGrabber(PatternDataGrabber):
    """Data grabber for SART (Sustained Attention to Response Task) paradigm.

    This data grabber extends PatternDataGrabber to automatically apply
    SART-specific processing (behavioral metadata extraction, channel filtering)
    to all EEG file types during data loading.
    """

    def __init__(
        self,
        datadir: Union[str, Path],
        types: Optional[List[str]] = None,
        patterns: Optional[Dict] = None,
        replacements: Optional[List[str]] = None,
        preserve_annotations: bool = True,
        extract_behavioral_params: bool = True,
        channel_selection: Union[str, List[str]] = "eeg",
        **kwargs,
    ):
        """Initialize SART Data Grabber.

        Parameters
        ----------
        datadir : str or Path
            Path to the data directory.
        types : list of str, optional
            Data types to grab. If None, defaults to ["EEG"].
        patterns : dict, optional
            Patterns for each data type. If None, uses default SART patterns.
        replacements : list of str, optional
            Replacement variables. If None, defaults to ["subject"].
        preserve_annotations : bool, default=True
            Whether to preserve epoch annotations in the epochs object.
        extract_behavioral_params : bool, default=True
            Whether to extract behavioral parameters from event descriptions.
        channel_selection : str or list, default='eeg'
            Which channels to keep ('all', 'eeg', or list of channel names).
        **kwargs
            Additional arguments passed to PatternDataGrabber.
        """
        # Set defaults for SART paradigm
        if types is None:
            types = ["EEG"]

        if patterns is None:
            patterns = {
                "EEG": {
                    "pattern": "{subject}_task-*_desc-*_epo.fif",
                    "space": "native",
                }
            }

        if replacements is None:
            replacements = ["subject"]

        # Store SART-specific parameters
        self.preserve_annotations = preserve_annotations
        self.extract_behavioral_params = extract_behavioral_params
        self.channel_selection = channel_selection

        super().__init__(
            datadir=datadir,
            types=types,
            patterns=patterns,
            replacements=replacements,
            **kwargs,
        )

    def __getitem__(self, element: Union[str, Dict]) -> Dict:
        """Get data for a specific element with SART processing.

        Parameters
        ----------
        element : str or dict
            Element identifier or element dictionary.

        Returns
        -------
        dict
            Dictionary containing processed EEG data with SART metadata.
        """
        # Get data using parent class (reads epochs file)
        data = super().__getitem__(element)

        # Store element information for metadata
        # Junifer requires 'element' key in meta for storage
        element_dict = (
            element if isinstance(element, dict) else {"element": element}
        )

        # Apply SART-specific processing using SARTDataReader
        # Note: When a datagrabber is present, the datareader in YAML is not used,
        # so we process the data here
        if "EEG" in data:
            from ..datareader.sart_datareader import SARTDataReader

            sart_datareader = SARTDataReader(
                preserve_annotations=self.preserve_annotations,
                extract_behavioral_params=self.extract_behavioral_params,
                channel_selection=self.channel_selection,
            )
            processed_data = sart_datareader._fit_transform(data)

            # Update the data dictionary with processed epochs
            if "EEG" in processed_data and "epochs" in processed_data["EEG"]:
                epochs = processed_data["EEG"]["epochs"]

                # Replace the data with the processed epochs object
                data["EEG"]["data"] = epochs

                # Add only JSON-serializable metadata
                # Don't include the epochs object itself or other complex objects
                data["EEG"]["meta"] = {
                    "element": element_dict,
                    "n_epochs": processed_data["EEG"]["n_epochs"],
                    "n_channels": processed_data["EEG"]["n_channels"],
                    "sfreq": float(processed_data["EEG"]["sfreq"]),
                    "tmin": float(processed_data["EEG"]["tmin"]),
                    "tmax": float(processed_data["EEG"]["tmax"]),
                    "channel_names": processed_data["EEG"]["channel_names"],
                }

                print(
                    f"[SART DATAGRABBER] Processed {len(epochs)} epochs "
                    f"with {len(epochs.ch_names)} channels"
                )
                if epochs.metadata is not None:
                    print(
                        f"[SART DATAGRABBER] Metadata attached: "
                        f"{list(epochs.metadata.columns)}"
                    )

            # CRITICAL: Remove 'path' key to prevent Junifer from reloading the file
            # and losing the processed annotations. The data is already loaded in memory.
            if "path" in data["EEG"]:
                del data["EEG"]["path"]

        return data
