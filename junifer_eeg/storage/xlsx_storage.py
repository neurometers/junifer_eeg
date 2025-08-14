"""Doctor-friendly XLSX storage for EEG markers."""

import hashlib
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Union

import numpy as np
import pandas as pd
from junifer.api.decorators import register_storage
from junifer.storage.base import BaseFeatureStorage


@register_storage
class XLSXFeatureStorage(BaseFeatureStorage):
    """XLSX storage for EEG markers with doctor-friendly formatting.

    This storage creates an Excel file with one sheet per marker, where:
    - Single values: Simple scalar value in clean format
    - Per-channel values: One column per channel with channel names
    - Per-epoch values: One row per epoch with epoch numbers
    - Matrices: Properly labeled rows and columns
    - Connectivity matrices: Channel names as row/column labels

    Each sheet includes:
    - Clear headers identifying what each row/column represents
    - Clean, minimal formatting focused on the actual marker values
    """

    def __init__(
        self,
        uri: str,
        single_output: bool = True,
        overwrite: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize XLSX storage.

        Parameters
        ----------
        uri : str
            Path to the XLSX file to create
        single_output : bool, default=True
            Whether to store all subjects in a single file
        overwrite : bool, default=False
            Whether to overwrite existing files
        **kwargs : Any
            Additional keyword arguments
        """
        # Initialize with supported storage types
        storage_types = ["vector", "matrix", "timeseries", "scalar_table"]
        super().__init__(
            uri=uri,
            storage_types=storage_types,
            single_output=single_output,
            **kwargs,
        )
        self.overwrite = overwrite
        self._data_buffer: Dict[str, Dict[str, Any]] = {}
        self._metadata_buffer: Dict[str, Dict[str, Any]] = {}

    def get_valid_inputs(self) -> list[str]:
        """Get valid storage types for input.

        Returns
        -------
        list[str]
            The list of storage types that can be used as input.
        """
        return ["vector", "matrix", "timeseries", "scalar_table"]

    def list_features(self) -> Dict[str, Dict[str, Any]]:
        """List the features in the storage.

        Returns
        -------
        Dict[str, Dict[str, Any]]
            Dictionary of features with MD5 keys and metadata values.
        """
        features = {}

        # Check if XLSX file exists and read it
        if Path(self.uri).exists():
            try:
                # Read Excel file to get sheet names (which are marker names)
                with pd.ExcelFile(self.uri) as xls:
                    for sheet_name in xls.sheet_names:
                        if sheet_name != "Overview":  # Skip overview sheet
                            # Create a simple MD5 hash for the feature
                            feature_md5 = hashlib.md5(
                                sheet_name.encode()
                            ).hexdigest()
                            features[feature_md5] = {
                                "name": sheet_name,
                                "type": "xlsx_marker",
                                "sheet_name": sheet_name,
                            }
            except Exception:
                # If file can't be read, return empty dict
                pass

        return features

    def read(
        self,
        feature_name: Optional[str] = None,
        feature_md5: Optional[str] = None,
    ) -> Dict[str, Union[str, list, np.ndarray]]:
        """Read stored feature.

        Parameters
        ----------
        feature_name : str, optional
            Name of the feature to read
        feature_md5 : str, optional
            MD5 hash of the feature to read

        Returns
        -------
        Dict[str, Union[str, list, np.ndarray]]
            The stored feature as a dictionary
        """
        if not Path(self.uri).exists():
            raise FileNotFoundError(f"XLSX file {self.uri} does not exist")

        # Determine sheet name to read
        sheet_name = feature_name
        if feature_md5 and not feature_name:
            # Look up feature name from MD5
            features = self.list_features()
            if feature_md5 in features:
                sheet_name = features[feature_md5]["sheet_name"]
            else:
                raise ValueError(f"Feature with MD5 {feature_md5} not found")

        if not sheet_name:
            raise ValueError(
                "Either feature_name or feature_md5 must be provided"
            )

        # Read the specific sheet
        try:
            df = pd.read_excel(self.uri, sheet_name=sheet_name)

            # Convert DataFrame to dictionary format
            result = {
                "data": df.to_numpy(),
                "columns": df.columns.tolist(),
                "sheet_name": sheet_name,
            }

            return result

        except Exception as e:
            raise ValueError(f"Error reading sheet {sheet_name}: {e}") from e

    def read_df(
        self,
        feature_name: Optional[str] = None,
        feature_md5: Optional[str] = None,
    ) -> pd.DataFrame:
        """Read feature into a pandas DataFrame.

        Parameters
        ----------
        feature_name : str, optional
            Name of the feature to read
        feature_md5 : str, optional
            MD5 hash of the feature to read

        Returns
        -------
        pd.DataFrame
            The features as a DataFrame
        """
        if not Path(self.uri).exists():
            raise FileNotFoundError(f"XLSX file {self.uri} does not exist")

        # Determine sheet name to read
        sheet_name = feature_name
        if feature_md5 and not feature_name:
            # Look up feature name from MD5
            features = self.list_features()
            if feature_md5 in features:
                sheet_name = features[feature_md5]["sheet_name"]
            else:
                raise ValueError(f"Feature with MD5 {feature_md5} not found")

        if not sheet_name:
            raise ValueError(
                "Either feature_name or feature_md5 must be provided"
            )

        # Read the specific sheet
        try:
            return pd.read_excel(self.uri, sheet_name=sheet_name)
        except Exception as e:
            raise ValueError(f"Error reading sheet {sheet_name}: {e}") from e

    def store_metadata(self, meta_md5: str, element: Dict, meta: Dict) -> None:
        """Store metadata.

        Parameters
        ----------
        meta_md5 : str
            The metadata MD5 hash
        element : Dict
            The element as a dictionary
        meta : Dict
            The metadata as a dictionary
        """
        # Store metadata in buffer for later use - include marker name extraction
        # The meta dict contains the full marker config, we need to extract just the name
        if isinstance(meta, dict) and "name" in meta:
            marker_name = meta["name"]
        elif isinstance(meta, dict) and "marker" in meta:
            marker_name = meta["marker"]
        else:
            marker_name = element.get("marker") or element.get("name")
        self._metadata_buffer[meta_md5] = {
            "element": element,
            "meta": meta,
            "marker_name": marker_name,
        }

    def store_matrix(
        self,
        meta_md5: str,
        element: dict,
        data: np.ndarray,
        col_names: Optional[Iterable[str]] = None,
        row_names: Optional[Iterable[str]] = None,
        matrix_kind: str = "full",
        diagonal: bool = True,
    ) -> None:
        """Store matrix data."""
        # Get the actual marker name from metadata buffer
        marker_name = None
        if meta_md5 in self._metadata_buffer:
            marker_name = self._metadata_buffer[meta_md5].get("marker_name")

        # Fallback to element or meta_md5 if not found
        if not marker_name:
            marker_name = element.get(
                "marker", element.get("name", f"matrix_{meta_md5[:8]}")
            )
        self._data_buffer[marker_name] = {
            "kind": "matrix",
            "data": data,
            "meta": element,
            "col_names": col_names,
            "row_names": row_names,
            "matrix_kind": matrix_kind,
            "diagonal": diagonal,
        }

    def store_vector(
        self,
        meta_md5: str,
        element: dict,
        data: Union[np.ndarray, list],
        col_names: Optional[Iterable[str]] = None,
    ) -> None:
        """Store vector data."""
        # Get the actual marker name from metadata buffer
        marker_name = None
        if meta_md5 in self._metadata_buffer:
            marker_name = self._metadata_buffer[meta_md5].get("marker_name")

        # Fallback to element or meta_md5 if not found
        if not marker_name:
            marker_name = element.get(
                "marker", element.get("name", f"vector_{meta_md5[:8]}")
            )

        self._data_buffer[marker_name] = {
            "kind": "vector",
            "data": data,
            "meta": element,
            "col_names": col_names,
        }
        # Write immediately for single_output mode
        if self.single_output:
            self._write_xlsx_file()

    def store_scalar_table(
        self,
        meta_md5: str,
        element: dict,
        data: np.ndarray,
        col_names: Optional[Iterable[str]] = None,
        row_names: Optional[Iterable[str]] = None,
        row_header_col_name: Optional[str] = "feature",
    ) -> None:
        """Store scalar table data."""
        # Get the actual marker name from metadata buffer
        marker_name = None
        if meta_md5 in self._metadata_buffer:
            marker_name = self._metadata_buffer[meta_md5].get("marker_name")

        # Fallback to element or meta_md5 if not found
        if not marker_name:
            marker_name = element.get(
                "marker", element.get("name", f"scalar_{meta_md5[:8]}")
            )
        self._data_buffer[marker_name] = {
            "kind": "scalar_table",
            "data": data,
            "meta": element,
            "col_names": col_names,
            "row_names": row_names,
            "row_header_col_name": row_header_col_name,
        }

    def store_timeseries(
        self,
        meta_md5: str,
        element: dict,
        data: np.ndarray,
        col_names: Optional[Iterable[str]] = None,
        row_names: Optional[Iterable[str]] = None,
    ) -> None:
        """Store timeseries data."""
        # Get the actual marker name from metadata buffer
        marker_name = None
        if meta_md5 in self._metadata_buffer:
            marker_name = self._metadata_buffer[meta_md5].get("marker_name")

        # Fallback to element or meta_md5 if not found
        if not marker_name:
            marker_name = element.get(
                "marker", element.get("name", f"timeseries_{meta_md5[:8]}")
            )
        self._data_buffer[marker_name] = {
            "kind": "timeseries",
            "data": data,
            "meta": element,
            "col_names": col_names,
            "row_names": row_names,
        }

    def _format_marker_data(
        self, marker_name: str, marker_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """Format marker data into a doctor-friendly DataFrame.

        Parameters
        ----------
        marker_name : str
            Name of the marker
        marker_data : Dict[str, Any]
            Dictionary containing data, meta, and element info for all subjects

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame ready for Excel export
        """
        # Collect all data across subjects
        all_data = []

        # Handle the actual data structure from our storage methods
        data = marker_data["data"]
        meta = marker_data.get("meta", {})

        # Get subject info from meta
        subject_id = meta.get("subject", "unknown_subject")

        # Extract the actual marker values based on data type
        if isinstance(data, dict):
            # Dictionary data (multiple values)
            for data_key, values in data.items():
                if isinstance(values, (int, float, np.number)):
                    # Single value per subject
                    all_data.append(
                        {
                            "Subject": subject_id,
                            "Marker": data_key,
                            "Value": float(values),
                        }
                    )

                elif isinstance(values, np.ndarray):
                    if values.ndim == 1:
                        # Vector (e.g., per-channel values)
                        for i, val in enumerate(values):
                            all_data.append(
                                {
                                    "Subject": subject_id,
                                    "Marker": data_key,
                                    "Channel/Index": f"Ch_{i + 1}",
                                    "Value": float(val),
                                }
                            )

                    elif values.ndim == 2:
                        # Matrix (e.g., connectivity matrix, time-frequency)
                        for i in range(values.shape[0]):
                            for j in range(values.shape[1]):
                                all_data.append(
                                    {
                                        "Subject": subject_id,
                                        "Marker": data_key,
                                        "Row": f"Ch_{i + 1}",
                                        "Column": f"Ch_{j + 1}",
                                        "Value": float(values[i, j]),
                                    }
                                )
                    else:
                        # Higher dimensional arrays - flatten with indices
                        flat_values = values.flatten()
                        indices = np.unravel_index(
                            range(len(flat_values)), values.shape
                        )
                        for idx, val in enumerate(flat_values):
                            coord_str = "_".join(
                                [
                                    f"Dim{d}_{indices[d][idx]}"
                                    for d in range(values.ndim)
                                ]
                            )
                            all_data.append(
                                {
                                    "Subject": subject_id,
                                    "Marker": data_key,
                                    "Coordinates": coord_str,
                                    "Value": float(val),
                                }
                            )

        if not all_data:
            # Return empty DataFrame with basic structure
            return pd.DataFrame(columns=["Subject", "Marker", "Value"])

        df = pd.DataFrame(all_data)

        # Add marker metadata as a header comment
        if marker_data:
            first_subject = next(iter(marker_data.values()))
            meta = first_subject.get("meta", {})

            # Create metadata summary
            metadata_rows = []
            metadata_rows.append(
                {
                    "Subject": "METADATA",
                    "Marker": "Marker Name",
                    "Value": marker_name,
                }
            )

            for key, value in meta.items():
                if key not in ["element", "data"]:  # Skip redundant info
                    metadata_rows.append(
                        {
                            "Subject": "METADATA",
                            "Marker": str(key),
                            "Value": str(value),
                        }
                    )

            # Add empty row separator
            metadata_rows.append({"Subject": "", "Marker": "", "Value": ""})

            # Combine metadata and data
            metadata_df = pd.DataFrame(metadata_rows)
            df = pd.concat([metadata_df, df], ignore_index=True)

        return df

    def _format_marker_data_properly(
        self, marker_name: str, marker_data: Dict[str, Any]
    ) -> pd.DataFrame:
        """Format marker data properly with dimensions as rows/columns.

        Parameters
        ----------
        marker_name : str
            Name of the marker (will be sheet name)
        marker_data : Dict[str, Any]
            Dictionary containing data, meta, kind info

        Returns
        -------
        pd.DataFrame
            Properly formatted DataFrame with dimensions as rows/columns
        """
        data = marker_data["data"]
        col_names = marker_data.get("col_names")
        row_names = marker_data.get("row_names")

        # Convert data to numpy array for consistent handling
        if isinstance(data, dict):
            # Handle dictionary data (e.g., from spectral markers)
            if len(data) == 1:
                # Single key-value pair, use the value
                data_array = np.asarray(next(iter(data.values())))
            else:
                # Multiple values - create a summary table
                summary_data = []
                for key, value in data.items():
                    if isinstance(value, (int, float, np.number)):
                        summary_data.append([key, float(value)])
                    elif isinstance(value, (list, np.ndarray)):
                        arr = np.asarray(value)
                        summary_data.append(
                            [
                                key,
                                f"Array shape: {arr.shape}, mean: {np.mean(arr):.6f}",
                            ]
                        )

                return pd.DataFrame(summary_data, columns=["Feature", "Value"])
        else:
            data_array = np.asarray(data)

        # Handle different data shapes
        if data_array.ndim == 0:
            # Scalar value
            return pd.DataFrame(
                {
                    "Channel/ROI": ["Row_0"],
                    marker_name: [float(data_array)],
                }
            )

        elif data_array.ndim == 1:
            # Vector data - create column format
            if col_names and len(col_names) == len(data_array):
                columns = list(col_names)
            else:
                columns = [f"Element_{i}" for i in range(len(data_array))]

            # Create DataFrame with single row
            df_data = {
                col: [float(val)] for col, val in zip(columns, data_array)
            }
            df = pd.DataFrame(df_data)

            return df

        elif data_array.ndim == 2:
            # Matrix data - rows and columns with proper labels
            if row_names and len(row_names) == data_array.shape[0]:
                index_names = list(row_names)
            else:
                index_names = [f"Row_{i}" for i in range(data_array.shape[0])]

            if col_names and len(col_names) == data_array.shape[1]:
                column_names = list(col_names)
            else:
                column_names = [f"Col_{i}" for i in range(data_array.shape[1])]

            # Create DataFrame with proper labels
            df = pd.DataFrame(
                data_array, index=index_names, columns=column_names
            )

            # Reset index to make row names a column
            df = df.reset_index()
            df.rename(columns={"index": "Channel/ROI"}, inplace=True)

            return df

        else:
            # Higher dimensional data - flatten to 2D
            reshaped = data_array.reshape(data_array.shape[0], -1)
            columns = [f"Feature_{i}" for i in range(reshaped.shape[1])]
            index_names = [f"Sample_{i}" for i in range(reshaped.shape[0])]

            df = pd.DataFrame(reshaped, index=index_names, columns=columns)
            df = df.reset_index()
            df.rename(columns={"index": "Sample"}, inplace=True)

            return df

    def _get_units(self, marker_name: str, data_key: str) -> str:
        """Get appropriate units for the marker data."""
        marker_lower = marker_name.lower()

        if "spectral" in marker_lower or "psd" in marker_lower:
            return "dB" if "dB" in data_key else "Power (μV²/Hz)"
        elif "entropy" in marker_lower:
            return "bits"
        elif "complexity" in marker_lower:
            return "compression ratio"
        elif "decoding" in marker_lower:
            return "accuracy (0-1)"
        elif "topography" in marker_lower or "contrast" in marker_lower:
            return "μV"
        elif "connectivity" in marker_lower or "mutual" in marker_lower:
            return "information (bits)"
        else:
            return "arbitrary units"

    def _get_description(self, marker_name: str, data_key: str) -> str:
        """Get human-readable description for the marker data."""
        marker_lower = marker_name.lower()

        if "spectral" in marker_lower:
            if "delta" in data_key:
                return "Delta band power (1-4 Hz) - deep sleep, unconscious processes"
            elif "theta" in data_key:
                return (
                    "Theta band power (4-8 Hz) - memory, emotion, navigation"
                )
            elif "alpha" in data_key:
                return "Alpha band power (8-12 Hz) - relaxed awareness, closed eyes"
            elif "beta" in data_key:
                return "Beta band power (12-30 Hz) - active thinking, concentration"
            elif "gamma" in data_key:
                return "Gamma band power (30+ Hz) - consciousness, binding"
            else:
                return "EEG frequency band power"
        elif "entropy" in marker_lower:
            return "Signal complexity measure - higher values indicate more complex patterns"
        elif "complexity" in marker_lower:
            return "Algorithmic complexity - higher values indicate less compressible signals"
        elif "decoding" in marker_lower:
            return "Classification accuracy between conditions - 1.0 = perfect, 0.5 = chance"
        elif "topography" in marker_lower:
            return "Event-related potential amplitude at specific time window"
        elif "contrast" in marker_lower:
            return (
                "Difference in brain activity between experimental conditions"
            )
        elif "connectivity" in marker_lower or "mutual" in marker_lower:
            return "Functional connectivity strength between brain regions"
        else:
            return "EEG-derived biomarker"

    def _create_overview_sheet(self, writer: pd.ExcelWriter) -> None:
        """Create an overview sheet explaining the results."""
        overview_data = []

        # Add general information
        overview_data.append(
            {
                "Section": "GENERAL INFO",
                "Item": "Analysis Type",
                "Value": "EEG Biomarker Analysis",
                "Description": "Automated extraction of brain activity markers from EEG data",
            }
        )

        overview_data.append(
            {
                "Section": "GENERAL INFO",
                "Item": "Generated By",
                "Value": "Junifer EEG Extension",
                "Description": "Open-source neuroimaging feature extraction pipeline",
            }
        )

        # Add marker summary
        for marker_name in self._data_buffer.keys():
            overview_data.append(
                {
                    "Section": "MARKERS",
                    "Item": marker_name,
                    "Value": f'See "{marker_name}" sheet',
                    "Description": self._get_marker_summary(marker_name),
                }
            )

        # Add interpretation guide
        overview_data.extend(
            [
                {
                    "Section": "INTERPRETATION",
                    "Item": "Single Values",
                    "Value": "One number per subject",
                    "Description": "Aggregated biomarker across all brain regions and time",
                },
                {
                    "Section": "INTERPRETATION",
                    "Item": "Per-Channel Values",
                    "Value": "One value per electrode",
                    "Description": "Spatial distribution of biomarker across brain regions",
                },
                {
                    "Section": "INTERPRETATION",
                    "Item": "Connectivity Matrices",
                    "Value": "Channel x Channel grid",
                    "Description": "Functional connectivity between brain regions",
                },
                {
                    "Section": "CLINICAL NOTES",
                    "Item": "Normal Ranges",
                    "Value": "Vary by age/condition",
                    "Description": "Compare with normative databases or control groups",
                },
                {
                    "Section": "CLINICAL NOTES",
                    "Item": "Statistical Significance",
                    "Value": "Requires group analysis",
                    "Description": "Individual values should be interpreted in clinical context",
                },
            ]
        )

        overview_df = pd.DataFrame(overview_data)
        overview_df.to_excel(writer, sheet_name="Overview", index=False)

        # Format overview sheet
        self._format_sheet(writer, "Overview", overview_df)

    def _get_marker_summary(self, marker_name: str) -> str:
        """Get clinical summary for each marker type."""
        marker_lower = marker_name.lower()

        if "spectral" in marker_lower or "psd" in marker_lower:
            return "Brain oscillation power in different frequency bands"
        elif "entropy" in marker_lower:
            return "Signal complexity - abnormal in neurological conditions"
        elif "complexity" in marker_lower:
            return "Information content - reduced in cognitive impairment"
        elif "decoding" in marker_lower:
            return "Brain state classification accuracy - cognitive processing measure"
        elif "topography" in marker_lower:
            return (
                "Event-related brain response - cognitive component analysis"
            )
        elif "contrast" in marker_lower:
            return "Difference in brain response between conditions"
        elif "connectivity" in marker_lower or "mutual" in marker_lower:
            return "Functional connectivity between brain regions"
        else:
            return "EEG-derived neurophysiological biomarker"

    def _format_sheet(
        self, writer: pd.ExcelWriter, sheet_name: str, df: pd.DataFrame
    ) -> None:
        """Apply formatting to make sheets more readable."""
        try:
            from openpyxl.styles import Alignment, Font, PatternFill

            worksheet = writer.sheets[sheet_name]

            # Style headers
            header_font = Font(bold=True, color="FFFFFF")
            header_fill = PatternFill(
                start_color="366092", end_color="366092", fill_type="solid"
            )

            # Apply header formatting
            for cell in worksheet[1]:  # First row
                cell.font = header_font
                cell.fill = header_fill
                cell.alignment = Alignment(horizontal="center")

            # Style metadata rows (if they exist)
            for _, row in enumerate(
                worksheet.iter_rows(min_row=2, max_row=10), 2
            ):
                if len(row) > 0 and row[0].value == "METADATA":
                    for cell in row:
                        cell.fill = PatternFill(
                            start_color="E6E6FA",
                            end_color="E6E6FA",
                            fill_type="solid",
                        )
                        cell.font = Font(italic=True)

            # Auto-adjust column widths
            for column in worksheet.columns:
                max_length = 0
                column_letter = column[0].column_letter

                for cell in column:
                    try:
                        if len(str(cell.value)) > max_length:
                            max_length = len(str(cell.value))
                    except Exception:
                        pass

                adjusted_width = min(max_length + 2, 50)  # Cap at 50 chars
                worksheet.column_dimensions[
                    column_letter
                ].width = adjusted_width

        except ImportError:
            # openpyxl styling not available, skip formatting
            pass

    def _write_xlsx_file(self) -> None:
        """Write all buffered data to XLSX file immediately."""
        if not self._data_buffer:
            return

        # Ensure output directory exists
        output_path = Path(self.uri)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create Excel writer
        with pd.ExcelWriter(self.uri, engine="openpyxl", mode="w") as writer:
            # Create overview sheet first
            self._create_overview_sheet(writer)

            # Create one sheet per marker
            for marker_name, marker_data in self._data_buffer.items():
                df = self._format_marker_data_properly(
                    marker_name, marker_data
                )

                # Truncate sheet name if too long (Excel limit is 31 chars)
                sheet_name = (
                    marker_name[:31] if len(marker_name) > 31 else marker_name
                )

                # Write to sheet
                df.to_excel(writer, sheet_name=sheet_name, index=False)

    def collect(self) -> None:
        """Write all buffered data to XLSX file."""
        self._write_xlsx_file()
        print(f"EEG results saved to: {self.uri}")
        print(f"Generated {len(self._data_buffer)} marker sheets")
