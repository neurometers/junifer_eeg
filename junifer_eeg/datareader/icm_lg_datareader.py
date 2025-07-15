"""ICM Local-Global Equipment-Specific Data Reader for EEG data.

This data reader provides comprehensive support for the ICM Local-Global paradigm
across multiple EEG equipment types, with sophisticated trigger processing and
equipment-specific configurations based on the next_icm implementation.
"""

from pathlib import Path
from typing import Any, ClassVar, Dict, List, Optional

import mne
import numpy as np
from junifer.api.decorators import register_datareader
from junifer.datareader import DefaultDataReader

# ICM LG Event ID mappings
ICM_LG_EVENT_ID = {
    "HSTD": 10,  # Hierarchical Standard
    "HDVT": 20,  # Hierarchical Deviant
    "LSGS": 30,  # Local Standard Global Standard
    "LSGD": 40,  # Local Standard Global Deviant
    "LDGS": 60,  # Local Deviant Global Standard
    "LDGD": 50,  # Local Deviant Global Deviant
}

# Arduino trigger mappings for different equipment
ARDUINO_TRIGGER_MAP = {
    0x40: "HSTD",  # 64
    0x48: "HDVT",  # 72
    0x80: "LSGS",  # 128
    0x98: "LDGD",  # 152
    0x90: "LSGD",  # 144
    0x88: "LDGS",  # 136
    # Inverted tones
    0x60: "HSTD",  # 96
    0x68: "HDVT",  # 104
    0xA0: "LSGS",  # 160
    0xB8: "LDGD",  # 184
    0xB0: "LSGD",  # 176
    0xA8: "LDGS",  # 168
}


class ICMEquipmentDetector:
    """Detects EEG equipment type from file extensions and patterns."""

    EQUIPMENT_PATTERNS: ClassVar[Dict[str, List[str]]] = {
        "egi": [".raw", ".mff", ".mff.zip"],
        "brainvision": [".vhdr"],
        "ant": [".cnt"],
        "gtec": [".hdf5"],
        "biosemi": [".bdf"],
        "matlab": [".mat"],
    }

    @classmethod
    def detect_equipment(cls, file_path: Path) -> str:
        """Detect equipment type from file path."""
        file_path = Path(file_path)

        # Check for compound extensions first (e.g., .mff.zip)
        if str(file_path).endswith(".mff.zip"):
            return "egi"

        # Check single extensions
        suffix = file_path.suffix.lower()
        for equipment, patterns in cls.EQUIPMENT_PATTERNS.items():
            if suffix in patterns:
                return equipment

        # Default fallback
        return "egi"


class ICMTriggerProcessor:
    """Processes triggers for different EEG equipment types based on next_icm."""

    def __init__(self, equipment_type: str):
        self.equipment_type = equipment_type

    def process_triggers(self, raw: mne.io.Raw, **kwargs) -> mne.io.Raw:
        """Process triggers based on equipment type."""
        if self.equipment_type == "egi":
            return self._process_egi_triggers(raw, **kwargs)
        if self.equipment_type == "brainvision":
            return self._process_brainvision_triggers(raw, **kwargs)
        if self.equipment_type == "ant":
            return self._process_ant_triggers(raw, **kwargs)
        if self.equipment_type == "biosemi":
            return self._process_biosemi_triggers(raw, **kwargs)
        return self._process_generic_triggers(raw, **kwargs)

    def _process_egi_triggers(self, raw: mne.io.Raw, **kwargs) -> mne.io.Raw:
        """Process EGI-specific triggers based on next_icm implementation."""
        # Ensure STI 014 channel exists
        if "STI 014" not in raw.ch_names:
            stim_data = np.zeros((1, len(raw.times)))
            info = mne.create_info(["STI 014"], raw.info["sfreq"], ["stim"])
            stim_raw = mne.io.RawArray(stim_data, info)
            raw.add_channels([stim_raw], force_update_info=True)

        # Check for digital trigger channels (DIN channels)
        dnames = [x for x in raw.ch_names if x.startswith("D")]
        if dnames:
            return self._process_digital_triggers(raw, dnames, **kwargs)

        # Check for hardware event channels
        return self._process_hardware_event_triggers(raw, **kwargs)

    def _process_digital_triggers(
        self,
        raw: mne.io.Raw,
        dnames: List[str],
        **kwargs,
    ) -> mne.io.Raw:
        """Process digital trigger channels (based on next_icm _read_lga_egi_generic)."""
        ttl_fix = kwargs.get("ttl_fix", "auto")

        # Auto-detect TTL fix necessity
        if ttl_fix == "auto":
            ttl_fix = True
            if "DIN3" in raw.ch_names:
                din3_idx = raw.ch_names.index("DIN3")
                if np.sum(raw._data[din3_idx, :]) > 10:
                    ttl_fix = False

        # Extract digital values
        values = [
            int(x.replace("DIN", "").replace("DI", "").replace("D", ""))
            for x in dnames
        ]

        if ttl_fix:
            values = 255 - np.array(values)

        # Initialize stimulus data
        stim_data = np.zeros_like(raw._data[raw.ch_names.index(dnames[0]), :])

        # Combine digital channels
        for dchan, dvalue in zip(dnames, values):
            if dvalue == 0:
                continue
            ddata = raw._data[raw.ch_names.index(dchan), :]
            idx = np.where(ddata != 0)[0]

            if kwargs.get("fix_reverse", False):
                dvalue = self._fix_reverse(dvalue)

            if dvalue >= 192:
                dvalue = 0
            stim_data[idx] = dvalue

        # Additional fix for bit reversal
        if kwargs.get("fix_reverse", False):
            idx = np.where(stim_data != 0)[0]
            fix_data = stim_data[idx]
            idx_128 = np.where(fix_data == 128)[0]
            fix_data[idx_128] = 0
            fix_data[idx_128 + 1] = 0
            stim_data[idx] = fix_data

        # Process trigger mapping
        stim_data = stim_data.astype(int)
        masked_data = stim_data & 0xF8
        new_trigger_data = np.zeros_like(masked_data)

        # Fix inconsistent values
        masked_data[masked_data < 0x3F] = 0

        # Fix consecutive samples with same trigger value
        repeated = np.logical_and(
            masked_data[:-1] != 0,
            np.ediff1d(masked_data) == 0,
        )
        repeated = np.where(repeated)[0] + 1
        masked_data[repeated] = 0

        # Find blocks
        idx = np.where(masked_data != 0)[0]
        if (
            len(idx) > 0
            and np.median(np.diff(idx)) / raw.info["sfreq"] <= 0.100
        ):
            # Group by 10 triggers (old arduino)
            idx = idx.reshape(-1, 10)[:, 0]

        # Map triggers to ICM LG events
        for block in idx:
            this_event_value = masked_data[block]
            if this_event_value in ARDUINO_TRIGGER_MAP:
                this_event = ICM_LG_EVENT_ID[
                    ARDUINO_TRIGGER_MAP[this_event_value]
                ]
                new_trigger_data[block] = this_event

        # Update stimulus channel
        stim_idx = raw.ch_names.index("STI 014")
        raw._data[stim_idx] = new_trigger_data

        # Drop digital channels
        raw.drop_channels(dnames)

        return raw

    def _process_hardware_event_triggers(
        self,
        raw: mne.io.Raw,
        **kwargs,
    ) -> mne.io.Raw:
        """Process hardware event triggers for EGI (based on next_icm _check_clean_trigger)."""
        # Check for hardware event channels
        has_hstd = any(k in raw.ch_names for k in ["HXX1", "HXX2"]) or any(
            k in raw.ch_names for k in ["HXY1", "HXY2"]
        )

        if not has_hstd:
            return raw

        # Determine condition type
        is_hstd = any(k in raw.ch_names for k in ["HXX1", "HXX2"]) and not any(
            k in raw.ch_names for k in ["HXY1", "HXY2"]
        )

        # Process trigger mapping
        trigger = raw._data[raw.ch_names.index("STI 014")]

        for k, v in raw.event_id.items():
            if k in ("HXX1", "HXX2"):
                new_val = ICM_LG_EVENT_ID["HSTD"]
            elif k in ("HXY1", "HXY2"):
                new_val = ICM_LG_EVENT_ID["HDVT"]
            elif k in ("XXX1", "XXX2") and is_hstd:
                new_val = ICM_LG_EVENT_ID["LSGS"]
            elif k in ("XXX1", "XXX2") and not is_hstd:
                new_val = ICM_LG_EVENT_ID["LSGD"]
            elif k in ("XXY1", "XXY2") and is_hstd:
                new_val = ICM_LG_EVENT_ID["LDGD"]
            elif k in ("XXY1", "XXY2") and not is_hstd:
                new_val = ICM_LG_EVENT_ID["LDGS"]
            else:
                new_val = 0

            value_index = np.where(trigger == v)
            trigger[value_index] = new_val

        # Drop old event channels
        drop_channels = [
            k for k in raw.ch_names if not k.startswith("E") and k != "STI 014"
        ]
        raw.drop_channels(drop_channels)

        return raw

    def _process_brainvision_triggers(
        self,
        raw: mne.io.Raw,
        **kwargs,
    ) -> mne.io.Raw:
        """Process BrainVision triggers (based on next_icm _read_lga_raw_bv)."""
        if "STI 014" not in raw.ch_names:
            return raw

        stim_chan_idx = raw.ch_names.index("STI 014")
        stim_data = raw._data[stim_chan_idx, :].astype(int) & 0xF8
        raw._data[stim_chan_idx, :] = stim_data.astype(float)

        events = mne.find_events(raw)
        new_data = np.zeros_like(raw._data[stim_chan_idx, :])
        to_use = events[::10]  # Every 10th event

        for key, value in ARDUINO_TRIGGER_MAP.items():
            idx = to_use[to_use[:, 2] == key, 0]
            new_data[idx] = ICM_LG_EVENT_ID[value]

        raw._data[stim_chan_idx, :] = new_data
        return raw

    def _process_ant_triggers(self, raw: mne.io.Raw, **kwargs) -> mne.io.Raw:
        """Process ANT triggers (based on next_icm _read_lga_cnt_ant)."""
        if "STI 014" not in raw.ch_names:
            return raw

        stim_idx = raw.ch_names.index("STI 014")
        stim_data = raw._data[stim_idx]
        new_stim_data = np.zeros_like(stim_data)

        for t_val in np.unique(stim_data).astype(int):
            if t_val == 0:
                continue
            t_type = ARDUINO_TRIGGER_MAP.get(t_val & 0xF8)
            if t_type:
                new_stim_data[stim_data == t_val] = ICM_LG_EVENT_ID[t_type]

        raw._data[stim_idx] = new_stim_data
        return raw

    def _process_biosemi_triggers(
        self,
        raw: mne.io.Raw,
        **kwargs,
    ) -> mne.io.Raw:
        """Process Biosemi triggers (based on next_icm _read_lga_bdf_bs)."""
        if "STI 014" not in raw.ch_names:
            return raw

        stim_idx = raw.ch_names.index("STI 014")
        stim_data = raw._data[stim_idx].astype(int)
        masked_data = stim_data & 0xF8
        new_trigger_data = np.zeros_like(masked_data)

        for value, kind in ARDUINO_TRIGGER_MAP.items():
            new_value = ICM_LG_EVENT_ID[kind]
            new_trigger_data[masked_data == value] = new_value

        raw._data[stim_idx] = new_trigger_data
        return raw

    def _process_generic_triggers(
        self,
        raw: mne.io.Raw,
        **kwargs,
    ) -> mne.io.Raw:
        """Process generic triggers."""
        if "STI 014" in raw.ch_names:
            events = mne.find_events(raw)
            if len(events) > 0:
                stim_idx = raw.ch_names.index("STI 014")
                stim_data = raw._data[stim_idx, :]
                new_stim_data = np.zeros_like(stim_data)

                for event_val in np.unique(events[:, 2]):
                    if event_val in ARDUINO_TRIGGER_MAP:
                        mapped_event = ICM_LG_EVENT_ID[
                            ARDUINO_TRIGGER_MAP[event_val]
                        ]
                        new_stim_data[stim_data == event_val] = mapped_event

                raw._data[stim_idx, :] = new_stim_data

        return raw

    @staticmethod
    def _fix_reverse(x: int) -> int:
        """Fix bit reversal (based on next_icm fix_reverse)."""
        if x == 0:
            return 0
        y = 0
        y += (1 - (x & 0x1)) << 7
        y += (x & 0x1) << 6
        y += (x & 0x2) << 4
        y += (x & 0x4) << 2
        y += x & 0x8
        y += (x & 0x10) >> 2
        y += (x & 0x20) >> 4
        y += (x & 0x20) >> 5
        return y


@register_datareader
class ICMLGDataReader(DefaultDataReader):
    """ICM Local-Global Equipment-Specific Data Reader.

    Provides comprehensive support for reading EEG data from multiple equipment types
    used in ICM Local-Global paradigm studies, based on the next_icm implementation.

    Parameters
    ----------
    equipment_type : str, optional
        Force specific equipment type. If None, auto-detects from file extension.
    apply_montage : bool, optional
        Whether to apply standard montage for the equipment. Default: True
    process_triggers : bool, optional
        Whether to process and map triggers to ICM LG event codes. Default: True
    trigger_params : dict, optional
        Additional parameters for trigger processing
    """

    def __init__(
        self,
        equipment_type: Optional[str] = None,
        apply_montage: bool = True,
        process_triggers: bool = True,
        trigger_params: Optional[Dict[str, Any]] = None,
    ):
        self.equipment_type = equipment_type
        self.apply_montage = apply_montage
        self.process_triggers = process_triggers
        self.trigger_params = trigger_params or {}

        super().__init__()

    def _fit_transform(
        self,
        input: Dict[str, Dict],
        params: Optional[Dict] = None,
    ) -> Dict:
        """Fit and transform EEG data with ICM LG specific processing."""
        # Use parent class for basic file reading
        output = super()._fit_transform(input, params)

        # Apply ICM LG specific processing to EEG data
        for data_type, data_info in output.items():
            if data_type == "EEG" and "data" in data_info:
                raw = data_info["data"]

                if isinstance(raw, mne.io.BaseRaw):
                    # Detect equipment type if not specified
                    if self.equipment_type is None:
                        file_path = data_info.get("path", "")
                        detected_equipment = (
                            ICMEquipmentDetector.detect_equipment(
                                Path(file_path),
                            )
                        )
                    else:
                        detected_equipment = self.equipment_type

                    # Process triggers if enabled
                    if self.process_triggers:
                        trigger_processor = ICMTriggerProcessor(
                            detected_equipment,
                        )
                        raw = trigger_processor.process_triggers(
                            raw,
                            **self.trigger_params,
                        )

                    # Apply equipment-specific configurations
                    raw = self._apply_equipment_config(raw, detected_equipment)

                    # Update the data
                    output[data_type]["data"] = raw
                    output[data_type]["equipment_type"] = detected_equipment

        return output

    def _apply_equipment_config(
        self,
        raw: mne.io.Raw,
        equipment_type: str,
    ) -> mne.io.Raw:
        """Apply equipment-specific configurations based on next_icm."""
        # Apply montage based on equipment type and channel count
        if self.apply_montage:
            n_eeg = sum(
                1
                for ch in raw.ch_names
                if ch.startswith("E")
                or ch in ["Fp1", "Fp2", "F3", "F4", "C3", "C4"]
            )

            try:
                if equipment_type == "egi":
                    if n_eeg >= 256:
                        montage = mne.channels.make_standard_montage(
                            "GSN-HydroCel-257",
                        )
                    elif n_eeg >= 128:
                        montage = mne.channels.make_standard_montage(
                            "GSN-HydroCel-129",
                        )
                    else:
                        montage = mne.channels.make_standard_montage(
                            "GSN-HydroCel-65",
                        )

                    # Drop Cz for EGI systems
                    if "Cz" in raw.ch_names:
                        raw.drop_channels(["Cz"])

                elif equipment_type == "biosemi":
                    montage = mne.channels.make_standard_montage("biosemi128")
                    # Drop external channels
                    ext_channels = [
                        "EXG1",
                        "EXG2",
                        "EXG3",
                        "EXG4",
                        "EXG5",
                        "EXG6",
                        "EXG7",
                        "EXG8",
                    ]
                    channels_to_drop = [
                        ch for ch in ext_channels if ch in raw.ch_names
                    ]
                    if channels_to_drop:
                        raw.drop_channels(channels_to_drop)

                else:
                    montage = mne.channels.make_standard_montage(
                        "standard_1020",
                    )

                raw.set_montage(montage, on_missing="ignore")

            except Exception as e:
                print(
                    f"Warning: Could not apply montage for {equipment_type}: {e}",
                )

        # Set description for equipment tracking
        n_final_eeg = len(
            [ch for ch in raw.ch_names if not ch.startswith("STI")],
        )
        raw.info["description"] = f"{equipment_type}/{n_final_eeg}"

        return raw
