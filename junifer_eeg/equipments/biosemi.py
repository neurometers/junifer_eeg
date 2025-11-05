"""Biosemi equipment ROI definitions.

This module provides ROI definitions for Biosemi recording systems,
adapted from the NICE pipeline. Currently supports 128-channel configuration.

References
----------
Adapted from NICE pipeline:
/site/bckp/next_icm/nice_ext/equipments/biosemi.py
"""

import numpy as np

from .rois import define_rois

# Biosemi 128-channel system ROI definitions (0-indexed)
_biosemi128_rois = {
    "p3a": np.array([0, 1, 32, 64, 75, 83, 84, 85, 88, 96, 110]),
    "p3b": np.array([0, 1, 3, 4, 18, 19, 31, 32, 64, 96, 110]),
    "mmn": np.array([0, 1, 32, 64, 75, 83, 84, 85, 88, 96, 110]),
    "cnv": np.array([0, 1, 32, 64, 75, 83, 84, 85, 88, 96, 110]),
    "Fz": np.array([75, 83, 84, 85, 88]),
    "Cz": np.array([0, 1, 32, 64, 96, 110]),
    "Pz": np.array([3, 4, 18, 19, 31]),
    "scalp": np.arange(128),
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


def register_biosemi_equipments():
    """Register all Biosemi equipment ROI definitions.

    This function should be called during module initialization to populate
    the ROI registry with Biosemi equipment definitions.
    """
    # Biosemi channel naming: A1-A32, B1-B32, C1-C32, D1-D32
    biosemi_ch_names = (
        [f"A{x}" for x in range(1, 33)]
        + [f"B{x}" for x in range(1, 33)]
        + [f"C{x}" for x in range(1, 33)]
        + [f"D{x}" for x in range(1, 33)]
    )

    define_rois("biosemi/128", _biosemi128_rois, ch_names=biosemi_ch_names)


# Auto-register on import
register_biosemi_equipments()
