"""Helper to convert old marker parameters to new hierarchical marker format."""

from typing import Any


def convert_legacy_params(params: dict[str, Any]) -> dict[str, Any]:
    """Convert legacy parameter names to new standardized names.

    This handles the renaming of aggregation parameters:
    - channel_aggregation_method -> channel_method
    - trial_aggregation_method -> trial_method

    Parameters
    ----------
    params : dict
        Parameters dict that may contain old parameter names.

    Returns
    -------
    dict
        Parameters dict with standardized parameter names.
    """
    new_params = params.copy()

    # Map old parameter names to new ones
    if "channel_aggregation_method" in new_params:
        new_params["channel_method"] = new_params.pop(
            "channel_aggregation_method"
        )

    if "trial_aggregation_method" in new_params:
        new_params["trial_method"] = new_params.pop("trial_aggregation_method")

    return new_params


def convert_to_spectral_power_bands_rois(
    old_params: dict[str, Any],
) -> dict[str, Any]:
    """Convert old SpectralPower parameters to SpectralPowerBands format.

    Parameters
    ----------
    old_params : dict
        Parameters from old SpectralPower marker.

    Returns
    -------
    dict
        Parameters for new SpectralPowerBands marker.
    """
    new_params = old_params.copy()

    # Map old parameter names to new ones
    if "channel_aggregation_method" in new_params:
        new_params["channel_method"] = new_params.pop(
            "channel_aggregation_method"
        )

    if "trial_aggregation_method" in new_params:
        new_params["trial_method"] = new_params.pop("trial_aggregation_method")

    # Convert fmin/fmax to bands if present
    if "fmin" in new_params or "fmax" in new_params:
        fmin = new_params.pop("fmin", None)
        fmax = new_params.pop("fmax", None)

        if fmin is not None and fmax is not None:
            # Create a band based on frequency range
            if (fmin, fmax) == (1.0, 4.0):
                band_name = "delta"
            elif (fmin, fmax) == (4.0, 8.0):
                band_name = "theta"
            elif (fmin, fmax) == (8.0, 12.0):
                band_name = "alpha"
            elif (fmin, fmax) == (12.0, 30.0):
                band_name = "beta"
            elif (fmin, fmax) == (30.0, 45.0):
                band_name = "gamma"
            elif (fmin, fmax) == (1.0, 45.0):
                band_name = "broadband"
            else:
                band_name = "custom"

            new_params["bands"] = {band_name: (fmin, fmax)}

    return new_params


def convert_to_permutation_entropy_rois(
    old_params: dict[str, Any],
) -> dict[str, Any]:
    """Convert old PermutationEntropy parameters to PermutationEntropy format.

    Parameters
    ----------
    old_params : dict
        Parameters from old PermutationEntropy marker.

    Returns
    -------
    dict
        Parameters for new PermutationEntropy marker.
    """
    new_params = old_params.copy()

    # Map old parameter names to new ones
    if "channel_aggregation_method" in new_params:
        new_params["channel_method"] = new_params.pop(
            "channel_aggregation_method"
        )

    if "trial_aggregation_method" in new_params:
        new_params["trial_method"] = new_params.pop("trial_aggregation_method")

    # Convert single tau to taus for new API
    if "tau" in new_params and "taus" not in new_params:
        tau = new_params.pop("tau")
        new_params["taus"] = (
            tau  # Keep as single value, constructor will handle conversion
        )

    # Convert single fmin/fmax to bands dict if needed
    if "fmin" in new_params or "fmax" in new_params:
        fmin = new_params.pop("fmin", None)
        fmax = new_params.pop("fmax", None)

        # If both are provided, create a single-band dict
        if fmin is not None and fmax is not None:
            # Create a band name based on frequency range
            if (fmin, fmax) == (4.0, 8.0):
                band_name = "theta"
            elif (fmin, fmax) == (8.0, 12.0):
                band_name = "alpha"
            elif (fmin, fmax) == (12.0, 30.0):
                band_name = "beta"
            elif (fmin, fmax) == (30.0, 45.0):
                band_name = "gamma"
            else:
                band_name = "custom"

            new_params["bands"] = {band_name: (fmin, fmax)}
        # If neither provided, don't set bands - use adaptive filter
    # If no fmin/fmax in params at all, don't set bands - use adaptive filter

    return new_params
