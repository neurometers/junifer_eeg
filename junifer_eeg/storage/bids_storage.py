"""BIDS-compliant feature storage for junifer_eeg.

This storage class formats output filenames according to BIDS conventions:
sub-<label>[_ses-<label>][_task-<label>][_acq-<label>]_desc-<label>_<datatype>.<ext>
"""

from pathlib import Path
from typing import Optional, Union

from junifer.api.decorators import register_storage
from junifer.storage import HDF5FeatureStorage
from junifer.utils import logger


def clean_entity_value(value: str, entity_name: str) -> str:
    """Clean entity value by removing BIDS prefixes if present.

    Parameters
    ----------
    value : str
        The entity value (e.g., "sub-01" or "01")
    entity_name : str
        The entity name (e.g., "subject", "session")

    Returns
    -------
    str
        Cleaned value without prefix (e.g., "01")
    """
    # Map entity names to their BIDS prefixes
    prefixes = {
        "subject": "sub-",
        "session": "ses-",
        "task": "task-",
        "acquisition": "acq-",
        "acq": "acq-",
        "run": "run-",
    }

    prefix = prefixes.get(entity_name, "")
    if prefix and value.startswith(prefix):
        return value[len(prefix) :]
    return value


def extract_bids_entities_from_element(
    element: dict[str, str],
) -> dict[str, str]:
    """Extract BIDS entities from element, parsing from file paths if needed.

    If session/task/acq aren't in element but are in a file path (like EEG path),
    extract them from the filename.

    Parameters
    ----------
    element : dict
        Element dictionary

    Returns
    -------
    dict
        Dictionary with extracted BIDS entities
    """
    import re
    from pathlib import Path as PathLib

    entities = {}

    # Get subject (clean it)
    subject = element.get("subject", "unknown")
    entities["subject"] = clean_entity_value(subject, "subject")

    # Try to get session, task, acq from element first
    entities["session"] = element.get("session", element.get("ses"))
    entities["task"] = element.get("task")
    entities["acq"] = element.get("acq", element.get("acquisition"))
    entities["run"] = element.get("run")

    # If missing, try to extract from file paths in element (like EEG path)
    if not all([entities["session"], entities["task"], entities["acq"]]):
        # Look for any path-like value in element (EEG, BOLD, etc.)
        for _, value in element.items():
            if isinstance(value, str) and ("/" in value or "\\" in value):
                # Extract BIDS entities from filename
                filename = PathLib(value).name

                # Extract session
                if not entities["session"]:
                    match = re.search(r"_ses-([a-zA-Z0-9]+)", filename)
                    if match:
                        entities["session"] = match.group(1)

                # Extract task
                if not entities["task"]:
                    match = re.search(r"_task-([a-zA-Z0-9]+)", filename)
                    if match:
                        entities["task"] = match.group(1)

                # Extract acquisition
                if not entities["acq"]:
                    match = re.search(r"_acq-([a-zA-Z0-9]+)", filename)
                    if match:
                        entities["acq"] = match.group(1)

                # Extract run
                if not entities["run"]:
                    match = re.search(r"_run-([a-zA-Z0-9]+)", filename)
                    if match:
                        entities["run"] = match.group(1)

                # If we found all, break
                if all(
                    [entities["session"], entities["task"], entities["acq"]]
                ):
                    break

    return entities


def element_to_bids_path(
    element: dict[str, str],
    base_dir: Path,
    base_name: str,
    suffix: str = "markers",
    extension: str = ".h5",
) -> Path:
    """Convert element dict to BIDS-compliant directory path and filename.

    Creates proper BIDS directory structure:
    sub-<label>/[ses-<label>/]eeg/sub-<label>[_ses-<label>]_task-<label>...<suffix>.<ext>

    Parameters
    ----------
    element : dict
        Element dictionary with keys like 'subject', 'session', 'task', 'acq', etc.
    base_dir : Path
        Base output directory
    base_name : str
        Base filename (will be used as desc-<base_name>)
    suffix : str, optional
        BIDS suffix (default "markers")
    extension : str, optional
        File extension including dot (default ".h5")

    Returns
    -------
    Path
        Full BIDS-compliant path including directories and filename

    Examples
    --------
    >>> element = {"subject": "sub-01", "session": "01", "task": "lg", "acq": "01"}
    >>> element_to_bids_path(element, Path("output"), "wsmi", "markers", ".h5")
    Path('output/sub-01/ses-01/eeg/sub-01_ses-01_task-lg_acq-01_desc-wsmi_markers.h5')
    """
    # Extract and clean BIDS entities
    entities = extract_bids_entities_from_element(element)
    subject = entities["subject"]

    # Build directory structure: sub-XX/[ses-YY/]eeg/
    path_parts = [base_dir, f"sub-{subject}"]

    # Add session directory if present
    if entities["session"]:
        path_parts.append(f"ses-{entities['session']}")

    # Add eeg directory
    path_parts.append("eeg")

    # Create directory path
    dir_path = Path(*path_parts)

    # Build filename following BIDS entity order
    filename_parts = []

    # Required: subject
    filename_parts.append(f"sub-{subject}")

    # Optional: session
    if entities["session"]:
        filename_parts.append(f"ses-{entities['session']}")

    # Optional: task
    if entities["task"]:
        filename_parts.append(f"task-{entities['task']}")

    # Optional: acquisition
    if entities["acq"]:
        filename_parts.append(f"acq-{entities['acq']}")

    # Optional: run
    if entities["run"]:
        filename_parts.append(f"run-{entities['run']}")

    # Optional: description
    if base_name:
        filename_parts.append(f"desc-{base_name}")

    # Add suffix (e.g., "markers", "eeg", "features")
    if suffix:
        filename_parts.append(suffix)

    # Combine and add extension
    filename = "_".join(filename_parts) + extension

    return dir_path / filename


@register_storage
class BIDSFeatureStorage(HDF5FeatureStorage):
    """BIDS-compliant feature storage.

    This storage class wraps HDF5FeatureStorage and formats filenames
    according to BIDS conventions. It automatically extracts BIDS entities
    from elements and constructs proper BIDS filenames.

    Parameters
    ----------
    uri : str or pathlib.Path
        The base path/directory for output files. The filename part (if provided)
        will be used as the desc-<label> field. For example:
        - "output/wsmi_features.h5" → desc-wsmi_features
        - "output/" → desc-markers (default)
    suffix : str, optional
        BIDS suffix (default "markers"). Common values: "markers", "features",
        "connectivity", "eeg", etc.
    single_output : bool, optional
        If False, will create one file per element with BIDS naming.
        If True, will create only one file (default False for BIDS compliance).
    output_pickle : bool, optional
        If True, will automatically create pickle (.pkl) versions of each HDF5
        file after storage is complete. Pickle files will have the same BIDS
        filename with .pkl extension (default False).
    delete_h5_after_pickle : bool, optional
        If True and output_pickle is True, will delete HDF5 files after
        converting to pickle (default False).
    overwrite : bool or "update", optional
        Whether to overwrite existing file (default "update").
    compression : {0-9}, optional
        Level of gzip compression (default 7).
    force_float32 : bool, optional
        Whether to cast float64 to float32 (default True).
    chunk_size : int, optional
        Chunk size for collecting data (default 100).

    Examples
    --------
    In YAML configuration:

    >>> storage:
    >>>   kind: BIDSFeatureStorage
    >>>   uri: output/derivatives/junifer-eeg/wsmi_features.h5
    >>>   suffix: markers
    >>>   single_output: false
    >>>   output_pickle: true
    >>>   delete_h5_after_pickle: false

    This will create files like:
    - sub-01_ses-01_task-lg_acq-01_desc-wsmi_features_markers.h5
    - sub-01_ses-01_task-lg_acq-01_desc-wsmi_features_markers.pkl
    - sub-02_ses-01_task-lg_acq-01_desc-wsmi_features_markers.h5
    - sub-02_ses-01_task-lg_acq-01_desc-wsmi_features_markers.pkl

    See Also
    --------
    HDF5FeatureStorage : The underlying storage implementation.

    Notes
    -----
    For BIDS compliance, it's recommended to use `single_output: false` and
    place outputs in a derivatives directory structure:
    dataset/derivatives/junifer-eeg/sub-XX/[ses-YY/]

    """

    def __init__(
        self,
        uri: Union[str, Path],
        suffix: str = "markers",
        single_output: bool = False,
        output_pickle: bool = False,
        delete_h5_after_pickle: bool = False,
        overwrite: Union[bool, str] = "update",
        compression: int = 7,
        force_float32: bool = True,
        chunk_size: int = 100,
    ) -> None:
        self.suffix = suffix
        self.output_pickle = output_pickle
        self.delete_h5_after_pickle = delete_h5_after_pickle
        self._h5_files_created = []  # Track created H5 files for pickle conversion

        # Extract base name from uri for desc- field
        uri_path = Path(uri)
        self.base_desc = uri_path.stem if uri_path.suffix else "markers"

        # For BIDS, we typically want the directory, not a specific filename
        if uri_path.suffix:  # If uri has extension, use parent directory
            self.output_dir = uri_path.parent
        else:  # If uri is a directory
            self.output_dir = uri_path

        super().__init__(
            uri=uri,
            single_output=single_output,
            overwrite=overwrite,
            compression=compression,
            force_float32=force_float32,
            chunk_size=chunk_size,
        )

    def _fetch_correct_uri_for_io(self, element: Optional[dict]) -> str:
        """Return BIDS-compliant URI based on element.

        Parameters
        ----------
        element : dict, optional
            The element as dictionary.

        Returns
        -------
        str
            BIDS-formatted URI for accessing metadata and data.

        Raises
        ------
        RuntimeError
            If ``element=None`` when ``single_output=False``.
        """
        if not self.single_output and element is None:
            from junifer.utils import raise_error

            raise_error(
                msg="`element` must be provided when `single_output=False`",
                klass=RuntimeError,
            )

        if not self.single_output and element is not None:
            # Generate BIDS-compliant path (directory structure + filename)
            bids_path = element_to_bids_path(
                element=element,
                base_dir=self.output_dir,
                base_name=self.base_desc,
                suffix=self.suffix,
                extension=self.uri.suffix,
            )

            # Create directory structure
            bids_path.parent.mkdir(parents=True, exist_ok=True)

            uri = str(bids_path)
            logger.debug(f"Using BIDS path: {bids_path}")

            # Track this file for later pickle conversion
            if uri not in self._h5_files_created:
                self._h5_files_created.append(uri)
        else:
            # Single output - use original uri
            uri = str(self.uri)
            if uri not in self._h5_files_created:
                self._h5_files_created.append(uri)

        return uri

    def _convert_to_pickle_if_needed(self) -> None:
        """Convert stored HDF5 files to pickle if output_pickle is enabled."""
        if not self.output_pickle:
            return

        from ..reader import read_h5

        logger.info("Converting HDF5 files to pickle format...")

        for h5_path in self._h5_files_created:
            h5_file = Path(h5_path)

            if not h5_file.exists():
                logger.warning(f"HDF5 file not found: {h5_path}, skipping...")
                continue

            # Create pickle filename (same name, different extension)
            pkl_path = h5_file.with_suffix(".pkl")

            logger.info(f"Converting {h5_file.name} to {pkl_path.name}")

            try:
                # Read HDF5 and dump to pickle
                reader = read_h5(h5_path)
                reader.dump_all_to_pkl(pkl_path)

                logger.info(f"Successfully created {pkl_path}")

                # Optionally delete original HDF5
                if self.delete_h5_after_pickle:
                    h5_file.unlink()
                    logger.info(f"Deleted original HDF5 file: {h5_file}")
            except Exception as e:
                logger.error(f"Failed to convert {h5_path} to pickle: {e}")

    def __del__(self):
        """Cleanup: convert to pickle if needed when storage object is destroyed."""
        try:
            self._convert_to_pickle_if_needed()
        except Exception as e:
            # Don't raise exceptions in __del__
            logger.error(f"Error during pickle conversion in cleanup: {e}")


def convert_h5_to_bids_pkl(
    h5_path: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    suffix: str = "markers",
    delete_h5: bool = False,
) -> Path:
    """Convert HDF5 file to BIDS-named pickle file.

    Parameters
    ----------
    h5_path : str or Path
        Path to HDF5 file.
    output_dir : str or Path, optional
        Output directory for pickle file. If None, uses same directory as h5_path.
    suffix : str, optional
        BIDS suffix (default "markers").
    delete_h5 : bool, optional
        Whether to delete original HDF5 file (default False).

    Returns
    -------
    Path
        Path to created pickle file.

    Examples
    --------
    >>> convert_h5_to_bids_pkl(
    ...     "output/sub-01_ses-01_task-lg_markers.h5",
    ...     suffix="markers"
    ... )
    PosixPath('output/sub-01_ses-01_task-lg_markers.pkl')
    """
    from ..reader import read_h5

    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    # Determine output path
    if output_dir is None:
        output_dir = h5_path.parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Create pickle filename (same name, different extension)
    pkl_filename = h5_path.stem + ".pkl"
    pkl_path = output_dir / pkl_filename

    logger.info(f"Converting {h5_path} to {pkl_path}")

    # Read HDF5 and dump to pickle
    reader = read_h5(h5_path)
    reader.dump_all_to_pkl(pkl_path)

    logger.info(f"Successfully created {pkl_path}")

    # Optionally delete original HDF5
    if delete_h5:
        h5_path.unlink()
        logger.info(f"Deleted original HDF5 file: {h5_path}")

    return pkl_path
