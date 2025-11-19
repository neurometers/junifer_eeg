"""SART (Sustained Attention to Response Task) Data Reader for EEG epochs data.

This data reader handles pre-epoched EEG data from SART experiments,
preserving rich behavioral annotations and epoch-level metadata.
"""

from typing import Any, Dict, List, Optional, Union

import mne
import pandas as pd
from junifer.api.decorators import register_datareader
from junifer.datareader import DefaultDataReader


@register_datareader
class SARTDataReader(DefaultDataReader):
    """SART Data Reader for pre-epoched EEG data with behavioral annotations.

    This reader handles EEG epochs files (.fif) from SART experiments,
    extracting and preserving:
    - Epoch-level behavioral annotations
    - Channel information
    - Event codes and descriptions
    - Metadata for easy access by markers

    Parameters
    ----------
    preserve_annotations : bool, default=True
        Whether to preserve epoch annotations in the epochs object
    extract_behavioral_params : bool, default=True
        Whether to extract behavioral parameters from event descriptions
    channel_selection : str or list, default='all'
        Which channels to keep ('all', 'eeg', or list of channel names)
    """

    def __init__(
        self,
        preserve_annotations: bool = True,
        extract_behavioral_params: bool = True,
        channel_selection: Union[str, List[str]] = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.preserve_annotations = preserve_annotations
        self.extract_behavioral_params = extract_behavioral_params
        self.channel_selection = channel_selection

    def _fit_transform(
        self,
        input: Dict[str, Dict],
        params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """Fit and transform SART EEG epochs data.

        Parameters
        ----------
        input : Dict[str, Dict]
            Input data dictionary
        params : Optional[Dict]
            Additional parameters

        Returns
        -------
        Dict[str, Any]
            Transformed data with epochs and metadata
        """
        # Get the EEG file path
        eeg_data = input["EEG"]
        file_path = eeg_data["path"]

        print(f"SART DATA READER: Loading epochs from {file_path}")

        # Load epochs
        epochs = mne.read_epochs(file_path, preload=True, verbose=False)

        # Set equipment metadata and montage (required for CSD)
        from .utils import detect_and_set_equipment

        detect_and_set_equipment(epochs)

        print(
            f"SART DATA READER: Loaded {len(epochs)} epochs, {len(epochs.ch_names)} channels"
        )
        print(
            f"SART DATA READER: Time range: {epochs.tmin:.3f} to {epochs.tmax:.3f} seconds"
        )
        print(f"SART DATA READER: Sampling rate: {epochs.info['sfreq']} Hz")

        # Apply channel selection
        epochs = self._apply_channel_selection(epochs)

        # Extract and process annotations
        epoch_annotations = self._extract_epoch_annotations(epochs)

        # Create behavioral metadata if requested
        behavioral_metadata = None
        if self.extract_behavioral_params:
            behavioral_metadata = self._extract_behavioral_metadata(
                epoch_annotations
            )

        # Store annotations in epochs object for marker access
        if self.preserve_annotations:
            epochs.metadata = pd.DataFrame(epoch_annotations)
            if behavioral_metadata is not None:
                # Merge behavioral parameters into metadata
                for key, values in behavioral_metadata.items():
                    epochs.metadata[key] = values

        # Create output dictionary
        output = {
            "epochs": epochs,
            "epoch_annotations": epoch_annotations,
            "behavioral_metadata": behavioral_metadata,
            "n_epochs": len(epochs),
            "n_channels": len(epochs.ch_names),
            "channel_names": epochs.ch_names,
            "sfreq": epochs.info["sfreq"],
            "tmin": epochs.tmin,
            "tmax": epochs.tmax,
        }

        print(f"SART DATA READER: Successfully processed {len(epochs)} epochs")
        if behavioral_metadata:
            print(
                f"SART DATA READER: Extracted {len(behavioral_metadata)} behavioral parameters"
            )

        return {"EEG": output}

    def _apply_channel_selection(self, epochs: mne.Epochs) -> mne.Epochs:
        """Apply channel selection based on configuration."""
        if self.channel_selection == "all":
            return epochs
        elif self.channel_selection == "eeg":
            # Keep only EEG channels, drop stimulus/auxiliary channels
            picks = mne.pick_types(
                epochs.info, eeg=True, stim=False, exclude="bads"
            )
            return epochs.pick(picks)
        elif isinstance(self.channel_selection, list):
            # Keep only specified channels
            channels_to_keep = [
                ch for ch in self.channel_selection if ch in epochs.ch_names
            ]
            if not channels_to_keep:
                raise ValueError(
                    f"None of the specified channels found in data: {self.channel_selection}"
                )
            return epochs.pick(channels_to_keep)
        else:
            raise ValueError(
                f"Invalid channel_selection: {self.channel_selection}"
            )

    def _extract_epoch_annotations(
        self, epochs: mne.Epochs
    ) -> List[Dict[str, Any]]:
        """Extract annotations for each epoch."""
        annotations = []

        # Get event information
        events = epochs.events
        event_id = epochs.event_id

        # Create reverse mapping from event code to description
        code_to_desc = {code: desc for desc, code in event_id.items()}

        for i, event in enumerate(events):
            event_time, _, event_code = event
            event_desc = code_to_desc.get(event_code, f"Unknown_{event_code}")

            annotation = {
                "epoch_index": int(i),  # Convert to Python int
                "epoch_id": f"epoch_{i:04d}",
                "event_time": int(
                    event_time
                ),  # Convert numpy int to Python int
                "event_code": int(
                    event_code
                ),  # Convert numpy int to Python int
                "event_description": event_desc,
                "original_description": event_desc,
            }

            annotations.append(annotation)

        return annotations

    def _extract_behavioral_metadata(
        self, annotations: List[Dict[str, Any]]
    ) -> Dict[str, List[Any]]:
        """Extract behavioral parameters from event descriptions."""
        behavioral_params = {
            "response_type": [],  # go/nogo
            "accuracy": [],  # correct/incorrect
            "valence": [],  # valence score
            "confidence": [],  # confidence score
            "reaction_time": [],  # time parameter
            "probe_id": [],  # probe identifier
            "trial_id": [],  # trial identifier
            "onoff": [],  # onoff parameter
            "selfother": [],  # selfother parameter
            "average": [],  # average parameter
        }

        for annotation in annotations:
            desc = annotation["event_description"]

            # Parse structured descriptions like:
            # 'go/correct/onoff8/selfother9/valence77/time84/confidence84/average45/-4/probe3/37'

            # Initialize with defaults
            params = dict.fromkeys(behavioral_params.keys())

            if isinstance(desc, str) and "/" in desc:
                parts = desc.split("/")

                for part in parts:
                    part = part.strip()

                    # Response type
                    if part in ["go", "nogo"]:
                        params["response_type"] = part

                    # Accuracy
                    elif part in ["correct", "incorrect"]:
                        params["accuracy"] = part

                    # Numeric parameters
                    elif part.startswith("valence") and len(part) > 7:
                        try:
                            params["valence"] = int(part[7:])
                        except ValueError:
                            pass

                    elif part.startswith("confidence") and len(part) > 10:
                        try:
                            params["confidence"] = int(part[10:])
                        except ValueError:
                            pass

                    elif part.startswith("time") and len(part) > 4:
                        try:
                            params["reaction_time"] = int(part[4:])
                        except ValueError:
                            pass

                    elif part.startswith("onoff") and len(part) > 5:
                        try:
                            params["onoff"] = int(part[5:])
                        except ValueError:
                            pass

                    elif part.startswith("selfother") and len(part) > 9:
                        try:
                            params["selfother"] = int(part[9:])
                        except ValueError:
                            pass

                    elif part.startswith("average") and len(part) > 7:
                        try:
                            params["average"] = int(part[7:])
                        except ValueError:
                            pass

                    elif part.startswith("probe") and len(part) > 5:
                        try:
                            params["probe_id"] = int(part[5:])
                        except ValueError:
                            pass

                    # Trial ID (usually last numeric part)
                    elif part.isdigit():
                        params["trial_id"] = int(part)

            # Add to behavioral parameters (convert numpy int to Python int for JSON serialization)
            for key in behavioral_params.keys():
                value = params[key]
                # Convert numpy types to Python native types for JSON serialization
                if value is not None and hasattr(value, "item"):
                    value = (
                        value.item()
                    )  # Convert numpy scalar to Python scalar
                behavioral_params[key].append(value)

        return behavioral_params

    def get_output_type(self, input_type: str) -> str:
        """Get output type for given input type."""
        return input_type  # Pass through the same type

    def get_item(self, input: Dict[str, Any]) -> Dict[str, Any]:
        """Get item from input with full processing."""
        print(
            f"🔍 SART DataReader get_item() called with input keys: {list(input.keys())}"
        )
        print(f"🔍 SART DataReader input type: {type(input)}")

        # Call _fit_transform to process the data and add metadata
        processed_data = self._fit_transform(input)

        # Return the processed data in the format expected by markers
        # The marker expects input["data"] to contain the epochs object
        if "EEG" in processed_data:
            eeg_data = processed_data["EEG"]
            if "epochs" in eeg_data:
                epochs = eeg_data["epochs"]
                print(
                    f"🔍 SART DataReader: Returning processed epochs with metadata: {epochs.metadata is not None}"
                )
                if epochs.metadata is not None:
                    print(
                        f"🔍 SART DataReader: Metadata shape: {epochs.metadata.shape}"
                    )
                    print(
                        f"🔍 SART DataReader: Metadata columns: {list(epochs.metadata.columns)}"
                    )

                # Return the processed epochs object as the main data
                return {
                    "data": epochs,
                    "space": input.get("EEG", {}).get("space", "native"),
                    "path": input.get("EEG", {}).get("path", ""),
                    "meta": eeg_data,  # Include all metadata for reference
                }

        print("🔍 SART DataReader: Falling back to original input")
        # Fallback to original behavior if processing fails
        return input
