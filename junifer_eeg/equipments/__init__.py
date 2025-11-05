"""Equipment and ROI definitions for EEG analysis.

This module provides equipment-specific ROI definitions adapted from the NICE
pipeline, enabling consistent channel selection and spatial filtering across
different EEG recording systems.
"""

# Import equipment modules to trigger auto-registration
from . import biosemi as _biosemi_module
from . import brainvision as _brainvision_module
from . import egi as _egi_module
from .rois import (
    define_rois,
    get_ch_names,
    get_roi,
    get_roi_ch_names,
    list_available_equipments,
    list_available_rois,
)

# Prevent modules from appearing as unused
del _biosemi_module, _brainvision_module, _egi_module

__all__ = [
    "define_rois",
    "get_ch_names",
    "get_roi",
    "get_roi_ch_names",
    "list_available_equipments",
    "list_available_rois",
]
