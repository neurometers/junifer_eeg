"""Utilities for data readers."""

import mne


def detect_and_set_equipment(raw_or_epochs):
    """Detect equipment from channels and set montage. Raise error if fails.

    Parameters
    ----------
    raw_or_epochs : mne.io.Raw or mne.Epochs
        MNE object

    Raises
    ------
    RuntimeError
        If montage cannot be set properly
    """
    ch_names = raw_or_epochs.ch_names

    # Detect equipment from channel names
    egi_channels = [
        ch for ch in ch_names if ch.startswith("E") and ch[1:].isdigit()
    ]

    if len(egi_channels) >= 200:
        equipment = "egi/256"
        montage_name = "GSN-HydroCel-256"
    elif len(egi_channels) >= 100:
        equipment = "egi/128"
        montage_name = "GSN-HydroCel-129"
    elif len(egi_channels) >= 50:
        equipment = "egi/64"
        montage_name = "GSN-HydroCel-65"
    elif any(ch in ch_names for ch in ["Fp1", "Fp2", "Fz", "Cz", "Pz", "Oz"]):
        # Standard 10-20 (check BEFORE Biosemi to avoid C3/C4 false positive)
        if len(ch_names) >= 100:
            equipment = "brainvision/128"
        else:
            equipment = "brainvision/32"
        montage_name = "standard_1020"
    elif any(
        ch[0] in ["A", "B", "C", "D"] and len(ch) == 3 and ch[1:].isdigit()
        for ch in ch_names
    ):
        # Biosemi: A1-A32, B1-B32, C1-C32, D1-D32 (exactly 3 chars)
        equipment = "biosemi/128"
        montage_name = "biosemi128"
    else:
        raise RuntimeError(
            f"Cannot detect equipment type from channel names: {ch_names[:5]}..."
        )

    # Set montage
    montage = mne.channels.make_standard_montage(montage_name)
    raw_or_epochs.set_montage(montage, on_missing="ignore")

    # Verify montage was set
    if (
        raw_or_epochs.info.get("dig") is None
        or len(raw_or_epochs.info.get("dig", [])) == 0
    ):
        raise RuntimeError(
            f"Failed to set montage '{montage_name}' for {equipment}. "
            "No digitization points found. Required for CSD computation."
        )

    # Store equipment in info
    raw_or_epochs.info["description"] = f"equipment={equipment}"
