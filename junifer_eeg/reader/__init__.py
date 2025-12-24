"""Simplified HDF5 reader for junifer_eeg markers.

Now that markers have fixed tensor structures with preserved dimensions,
reading is straightforward: map marker class → dimension order.

Usage
-----
>>> from junifer_eeg.reader import read_h5
>>> reader = read_h5("output.h5")
>>> data = reader.get("psd_alpha")
>>> print(data.shape)
(5, 100, 64)  # bands x epochs x channels
>>> print(data.dim_names)
['bands', 'epochs', 'channels']
"""

from .reader import MarkerData, SimplifiedH5Reader, read_h5

__all__ = [
    "MarkerData",
    "SimplifiedH5Reader",
    "read_h5",
]
