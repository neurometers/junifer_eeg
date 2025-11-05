"""EGI equipment ROI definitions.

This module provides ROI definitions for various EGI recording systems,
adapted from the NICE pipeline. All definitions use 0-indexed channel numbers.

ROI Definitions
---------------
- **scalp**: All EEG electrodes on the scalp (excludes reference/aux channels)
- **nonscalp**: Reference and auxiliary channels
- **cnv**: Contingent Negative Variation ROI (frontal-central)
- **mmn**: Mismatch Negativity ROI (frontocentral)
- **p3a**: P3a component ROI (frontocentral)
- **p3b**: P3b component ROI (centroparietal)
- **Fz, Cz, Pz**: Midline electrode clusters
- **Fp1, Fp2, F3, F4, C3, C4, P3, P4, T5, T6, Oz**: Standard 10-20 positions

References
----------
Adapted from NICE pipeline:
/site/bckp/next_icm/nice_ext/equipments/egi.py
"""

import numpy as np

from .rois import define_rois

# EGI 256-channel system ROI definitions
# Indices are 0-based (NICE uses 1-based with "- 1" conversion)
_egi256_rois = {
    "p3a": np.array([6, 7, 9, 14, 15, 16, 22, 23, 45, 81, 132, 186]) - 1,
    "p3b": np.array([9, 45, 81, 100, 101, 110, 119, 128, 129, 132, 186]) - 1,
    "mmn": np.array([6, 7, 9, 14, 15, 16, 22, 23, 45, 81, 132, 186]) - 1,
    "cnv": np.array([6, 7, 14, 15, 16, 22, 23]) - 1,
    "Fz": np.array([6, 7, 14, 15, 16, 22, 23]) - 1,
    "Cz": np.array([9, 45, 81, 132, 186]) - 1,
    "Pz": np.array([100, 101, 110, 119, 128, 129]) - 1,
    "scalp": np.arange(224),  # E1-E224 (E225-E256 are non-scalp)
    "nonscalp": np.arange(224, 256),  # E225-E256
    "Fp1": np.array([26, 27, 32, 33, 34, 37, 38]) - 1,
    "Fp2": np.array([11, 12, 18, 19, 20, 25, 26]) - 1,
    "F3": np.array([30, 36, 40, 41, 42, 49, 50]) - 1,
    "F4": np.array([205, 206, 213, 214, 215, 223, 224]) - 1,
    "C3": np.array([51, 52, 58, 59, 60, 65, 66]) - 1,
    "C4": np.array([155, 164, 182, 183, 184, 195, 196]) - 1,
    "P3": np.array([76, 77, 85, 86, 87, 97, 98]) - 1,
    "P4": np.array([152, 153, 161, 162, 163, 171, 172]) - 1,
    "T5": np.array([83, 84, 85, 94, 95, 96, 104, 105, 106]) - 1,
    "T6": np.array([169, 170, 171, 177, 178, 179, 189, 190, 191]) - 1,
    "Oz": np.array([125, 136, 137, 138, 148]) - 1,
}

# EGI 128-channel system ROI definitions
_egi128_rois = {
    "p3a": np.array([4, 5, 11, 12, 16, 19, 7, 31, 55, 80, 106]) - 1,
    "p3b": np.array([7, 31, 55, 80, 106, 61, 62, 72, 78]) - 1,
    "mmn": np.array([4, 5, 11, 12, 16, 19, 7, 31, 55, 80, 106]) - 1,
    "cnv": np.array([4, 5, 11, 12, 16, 19]) - 1,
    "Fz": np.array([4, 5, 11, 12, 16, 19]) - 1,
    "Cz": np.array([7, 31, 55, 80, 106]) - 1,
    "Pz": np.array([61, 62, 72, 78]) - 1,
    "scalp": np.arange(124),  # E1-E124 (E125-E128 are non-scalp)
    "nonscalp": np.arange(124, 127),  # E125-E128 (note: only 3 channels)
    "Fp1": np.array([]),
    "Fp2": np.array([]),
    "F3": np.array([]),
    "F4": np.array([]),
    "C3": np.array([]),
    "C4": np.array([]),
    "P3": np.array([]),
    "P4": np.array([]),
    "T5": np.array([]),
    "T6": np.array([]),
    "Oz": np.array([]),
}

# EGI 64-channel system ROI definitions
_egi64_rois = {
    "p3a": np.array([3, 6, 8, 9, 4, 7, 16, 21, 41, 51, 54]) - 1,
    "p3b": np.array([4, 7, 16, 21, 41, 51, 54, 34, 33, 36, 38]) - 1,
    "mmn": np.array([3, 6, 8, 9, 4, 7, 16, 21, 41, 51, 54]) - 1,
    "cnv": np.array([3, 6, 8, 9]) - 1,
    "Fz": np.array([3, 6, 8, 9]) - 1,
    "Cz": np.array([4, 7, 16, 21, 41, 51, 54]) - 1,
    "Pz": np.array([34, 33, 36, 38]) - 1,
    "scalp": np.arange(64),  # All 64 channels are scalp
    "nonscalp": np.array([]),  # No non-scalp channels
    "Fp1": np.array([]),
    "Fp2": np.array([]),
    "F3": np.array([]),
    "F4": np.array([]),
    "C3": np.array([]),
    "C4": np.array([]),
    "P3": np.array([]),
    "P4": np.array([]),
    "T5": np.array([]),
    "T6": np.array([]),
    "Oz": np.array([]),
}

# EGI 257-channel system ROI definitions (256 + 1 vertex reference)
_egi257_rois = {
    "p3a": np.array([6, 7, 9, 14, 15, 16, 22, 23, 45, 81, 132, 186]) - 1,
    "p3b": np.array([9, 45, 81, 100, 101, 110, 119, 128, 129, 132, 186]) - 1,
    "mmn": np.array([6, 7, 9, 14, 15, 16, 22, 23, 45, 81, 132, 186]) - 1,
    "cnv": np.array([6, 7, 14, 15, 16, 22, 23]) - 1,
    "Fz": np.array([6, 7, 14, 15, 16, 22, 23]) - 1,
    "Cz": np.array([9, 45, 81, 132, 186]) - 1,
    "Pz": np.array([100, 101, 110, 119, 128, 129]) - 1,
    "scalp": np.arange(256),  # E1-E256 are scalp channels
    "nonscalp": np.array([256]),  # Vertex Reference is non-scalp
    "Fp1": np.array([26, 27, 32, 33, 34, 37, 38]) - 1,
    "Fp2": np.array([11, 12, 18, 19, 20, 25, 26]) - 1,
    "F3": np.array([30, 36, 40, 41, 42, 49, 50]) - 1,
    "F4": np.array([205, 206, 213, 214, 215, 223, 224]) - 1,
    "C3": np.array([51, 52, 58, 59, 60, 65, 66]) - 1,
    "C4": np.array([155, 164, 182, 183, 184, 195, 196]) - 1,
    "P3": np.array([76, 77, 85, 86, 87, 97, 98]) - 1,
    "P4": np.array([152, 153, 161, 162, 163, 171, 172]) - 1,
    "T5": np.array([83, 84, 85, 94, 95, 96, 104, 105, 106]) - 1,
    "T6": np.array([169, 170, 171, 177, 178, 179, 189, 190, 191]) - 1,
    "Oz": np.array([125, 136, 137, 138, 148]) - 1,
}

# Averaged montage ROI definitions (spatial averaging, minimal event-related ROIs)
_egi64a_rois = {
    "p3a": np.array([]),
    "p3b": np.array([]),
    "mmn": np.array([]),
    "cnv": np.array([]),
    "Fz": np.array([]),
    "Cz": np.array([]),
    "Pz": np.array([]),
    "scalp": np.arange(64),
    "nonscalp": np.array([]),
    "Fp1": np.array([]),
    "Fp2": np.array([]),
    "F3": np.array([]),
    "F4": np.array([]),
    "C3": np.array([]),
    "C4": np.array([]),
    "P3": np.array([]),
    "P4": np.array([]),
    "T5": np.array([]),
    "T6": np.array([]),
    "Oz": np.array([]),
}

_egi32a_rois = {
    "p3a": np.array([]),
    "p3b": np.array([]),
    "mmn": np.array([]),
    "cnv": np.array([]),
    "Fz": np.array([]),
    "Cz": np.array([]),
    "Pz": np.array([]),
    "scalp": np.arange(32),
    "nonscalp": np.array([]),
    "Fp1": np.array([]),
    "Fp2": np.array([]),
    "F3": np.array([]),
    "F4": np.array([]),
    "C3": np.array([]),
    "C4": np.array([]),
    "P3": np.array([]),
    "P4": np.array([]),
    "T5": np.array([]),
    "T6": np.array([]),
    "Oz": np.array([]),
}

_egi16a_rois = {
    "p3a": np.array([]),
    "p3b": np.array([]),
    "mmn": np.array([]),
    "cnv": np.array([]),
    "Fz": np.array([]),
    "Cz": np.array([]),
    "Pz": np.array([]),
    "scalp": np.arange(16),
    "nonscalp": np.array([]),
    "Fp1": np.array([]),
    "Fp2": np.array([]),
    "F3": np.array([]),
    "F4": np.array([]),
    "C3": np.array([]),
    "C4": np.array([]),
    "P3": np.array([]),
    "P4": np.array([]),
    "T5": np.array([]),
    "T6": np.array([]),
    "Oz": np.array([]),
}

_egi8a_rois = {
    "p3a": np.array([]),
    "p3b": np.array([]),
    "mmn": np.array([]),
    "cnv": np.array([]),
    "Fz": np.array([]),
    "Cz": np.array([]),
    "Pz": np.array([]),
    "scalp": np.arange(8),
    "nonscalp": np.array([]),
    "Fp1": np.array([]),
    "Fp2": np.array([]),
    "F3": np.array([]),
    "F4": np.array([]),
    "C3": np.array([]),
    "C4": np.array([]),
    "P3": np.array([]),
    "P4": np.array([]),
    "T5": np.array([]),
    "T6": np.array([]),
    "Oz": np.array([]),
}

_egi4a_rois = {
    "p3a": np.array([]),
    "p3b": np.array([]),
    "mmn": np.array([]),
    "cnv": np.array([]),
    "Fz": np.array([]),
    "Cz": np.array([]),
    "Pz": np.array([]),
    "scalp": np.arange(4),
    "nonscalp": np.array([]),
    "Fp1": np.array([]),
    "Fp2": np.array([]),
    "F3": np.array([]),
    "F4": np.array([]),
    "C3": np.array([]),
    "C4": np.array([]),
    "P3": np.array([]),
    "P4": np.array([]),
    "T5": np.array([]),
    "T6": np.array([]),
    "Oz": np.array([]),
}

_egi2a_rois = {
    "p3a": np.array([]),
    "p3b": np.array([]),
    "mmn": np.array([]),
    "cnv": np.array([]),
    "Fz": np.array([]),
    "Cz": np.array([]),
    "Pz": np.array([]),
    "scalp": np.arange(2),
    "nonscalp": np.array([]),
    "Fp1": np.array([]),
    "Fp2": np.array([]),
    "F3": np.array([]),
    "F4": np.array([]),
    "C3": np.array([]),
    "C4": np.array([]),
    "P3": np.array([]),
    "P4": np.array([]),
    "T5": np.array([]),
    "T6": np.array([]),
    "Oz": np.array([]),
}


def register_egi_equipments():
    """Register all EGI equipment ROI definitions.

    This function should be called during module initialization to populate
    the ROI registry with EGI equipment definitions.
    """
    # Generate EGI channel names (E1, E2, E3, ...)
    define_rois(
        "egi/256", _egi256_rois, ch_names=[f"E{i}" for i in range(1, 257)]
    )
    define_rois(
        "egi/128", _egi128_rois, ch_names=[f"E{i}" for i in range(1, 129)]
    )
    define_rois(
        "egi/64", _egi64_rois, ch_names=[f"E{i}" for i in range(1, 65)]
    )
    define_rois(
        "egi/257", _egi257_rois, ch_names=[f"E{i}" for i in range(1, 258)]
    )
    define_rois(
        "egi/64a", _egi64a_rois, ch_names=[f"E{i}" for i in range(1, 65)]
    )
    define_rois(
        "egi/32a", _egi32a_rois, ch_names=[f"E{i}" for i in range(1, 33)]
    )
    define_rois(
        "egi/16a", _egi16a_rois, ch_names=[f"E{i}" for i in range(1, 17)]
    )
    define_rois("egi/8a", _egi8a_rois, ch_names=[f"E{i}" for i in range(1, 9)])
    define_rois("egi/4a", _egi4a_rois, ch_names=[f"E{i}" for i in range(1, 5)])
    define_rois("egi/2a", _egi2a_rois, ch_names=[f"E{i}" for i in range(1, 3)])


# Auto-register on import
register_egi_equipments()
