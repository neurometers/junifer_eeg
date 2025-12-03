"""Helper to convert old marker parameters to new hierarchical marker format."""


def convert_spectral_power_params(old_params):
    """Convert old SpectralPower parameters to SpectralPowerBandsROIs format.

    Parameters
    ----------
    old_params : dict
        Parameters for old SpectralPower marker.

    Returns
    -------
    dict
        Parameters for new SpectralPowerBandsROIs marker.
    """
    new_params = old_params.copy()

    # Map old parameter names to new ones
    if "channel_aggregation_method" in new_params:
        channel_method = new_params.pop("channel_aggregation_method")
        if channel_method == "trim_mean80":
            new_params["channel_method"] = "trim_mean"
            if "channel_method_params" not in new_params:
                new_params["channel_method_params"] = {"proportiontocut": 0.1}
        else:
            new_params["channel_method"] = channel_method

    if "trial_aggregation_method" in new_params:
        trial_method = new_params.pop("trial_aggregation_method")
        if trial_method == "trim_mean80":
            new_params["trial_method"] = "trim_mean"
            if "trial_method_params" not in new_params:
                new_params["trial_method_params"] = {"proportiontocut": 0.1}
        else:
            new_params["trial_method"] = trial_method

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


def convert_permutation_entropy_params(old_params):
    """Convert old PermutationEntropy parameters to PermutationEntropyROIs format.

    Parameters
    ----------
    old_params : dict
        Parameters for old PermutationEntropy marker.

    Returns
    -------
    dict
        Parameters for new PermutationEntropyROIs marker.
    """
    new_params = old_params.copy()

    # Map old parameter names to new ones
    if "channel_aggregation_method" in new_params:
        channel_method = new_params.pop("channel_aggregation_method")
        if channel_method == "trim_mean80":
            new_params["channel_method"] = "trim_mean"
            if "channel_method_params" not in new_params:
                new_params["channel_method_params"] = {"proportiontocut": 0.1}
        else:
            new_params["channel_method"] = channel_method

    if "trial_aggregation_method" in new_params:
        trial_method = new_params.pop("trial_aggregation_method")
        if trial_method == "trim_mean80":
            new_params["trial_method"] = "trim_mean"
            if "trial_method_params" not in new_params:
                new_params["trial_method_params"] = {"proportiontocut": 0.1}
        else:
            new_params["trial_method"] = trial_method

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
