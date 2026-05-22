
import os
from pathlib import Path
import pydicom
import pandas as pd
import numpy as np

# =====================================================
# CONFIG
# =====================================================

BASE_FOLDER = r"C:\Users\efi015\OneDrive - University of Bergen\MRI_H2\CH4_Characterization2"

TARGET_PROTOCOLS = [
    "CH4_MSME_SAG",
    "CH4_RAREVTR_AX",
    "CH4_MSME_AX",
]

OUTPUT_EXCEL = "CH4_protocol_summary.xlsx"


# =====================================================
# FUNCTIONS
# =====================================================

def safe_get(ds, tag_name, default=None):
    return getattr(ds, tag_name, default)


def to_float(value):
    if value is None:
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def find_dicom_files(base_folder):
    dicom_files = []
    for root, _, files in os.walk(base_folder):
        for file in files:
            if file.lower().endswith(".dcm") or file.lower().startswith("mrim"):
                dicom_files.append(Path(root) / file)
    return sorted(dicom_files)


def read_metadata(base_folder, target_protocols):
    rows = []

    for path in find_dicom_files(base_folder):
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue

        protocol_name = safe_get(ds, "ProtocolName", "")
        series_description = safe_get(ds, "SeriesDescription", "")
        sequence_name = safe_get(ds, "SequenceName", "")

        # Main filter: ProtocolName
        if protocol_name not in target_protocols:
            continue

        pixel_spacing = safe_get(ds, "PixelSpacing", [np.nan, np.nan])

        rows.append({
            "ProtocolName": protocol_name,
            "SeriesDescription": series_description,
            "SequenceName": sequence_name,
            "File": path.name,
            "Folder": str(path.parent),
            "EchoTime_ms": to_float(safe_get(ds, "EchoTime")),
            "RepetitionTime_ms": to_float(safe_get(ds, "RepetitionTime")),
            "FlipAngle_deg": to_float(safe_get(ds, "FlipAngle")),
            "NumberOfAverages": to_float(safe_get(ds, "NumberOfAverages")),
            "Rows": safe_get(ds, "Rows"),
            "Columns": safe_get(ds, "Columns"),
            "PixelSpacing_row_mm": to_float(pixel_spacing[0]) if pixel_spacing is not None else np.nan,
            "PixelSpacing_col_mm": to_float(pixel_spacing[1]) if pixel_spacing is not None else np.nan,
            "SliceThickness_mm": to_float(safe_get(ds, "SliceThickness")),
            "InstanceNumber": safe_get(ds, "InstanceNumber"),
            "AcquisitionTime": safe_get(ds, "AcquisitionTime"),
        })

    return pd.DataFrame(rows)


def unique_sorted(series):
    values = pd.to_numeric(series, errors="coerce").dropna().unique()
    return sorted(values)


def format_values(values):
    return ", ".join(f"{v:g}" for v in values)


def estimate_spacing(values):
    if len(values) < 2:
        return np.nan
    return float(np.median(np.diff(sorted(values))))


def make_summary(metadata_df):
    summary_rows = []

    for protocol, group in metadata_df.groupby("ProtocolName"):
        te_values = unique_sorted(group["EchoTime_ms"])
        tr_values = unique_sorted(group["RepetitionTime_ms"])

        rows = group["Rows"].dropna().unique()
        cols = group["Columns"].dropna().unique()
        matrix = f"{rows[0]} × {cols[0]}" if len(rows) and len(cols) else ""

        px_row = group["PixelSpacing_row_mm"].dropna().unique()
        px_col = group["PixelSpacing_col_mm"].dropna().unique()
        slice_thick = group["SliceThickness_mm"].dropna().unique()

        resolution = ""
        if len(px_row) and len(px_col) and len(slice_thick):
            resolution = f"{px_row[0]:g} × {px_col[0]:g} × {slice_thick[0]:g}"

        summary_rows.append({
            "ProtocolName": protocol,
            "SequenceName": group["SequenceName"].dropna().iloc[0] if group["SequenceName"].notna().any() else "",
            "Number of DICOM files": len(group),
            "Number of unique TE values": len(te_values),
            "First TE [ms]": min(te_values) if te_values else np.nan,
            "Last TE [ms]": max(te_values) if te_values else np.nan,
            "Echo spacing [ms]": estimate_spacing(te_values),
            "TE values [ms]": format_values(te_values),
            "Number of unique TR values": len(tr_values),
            "TR values [ms]": format_values(tr_values),
            "Flip angle [deg]": group["FlipAngle_deg"].dropna().iloc[0] if group["FlipAngle_deg"].notna().any() else np.nan,
            "Averages": group["NumberOfAverages"].dropna().iloc[0] if group["NumberOfAverages"].notna().any() else np.nan,
            "Matrix": matrix,
            "Resolution [mm3]": resolution,
        })

    return pd.DataFrame(summary_rows)


# =====================================================
# RUN
# =====================================================

metadata_df = read_metadata(BASE_FOLDER, TARGET_PROTOCOLS)

if metadata_df.empty:
    raise RuntimeError("No DICOM files found with the selected ProtocolName values.")

summary_df = make_summary(metadata_df)

print(summary_df.to_string(index=False))

with pd.ExcelWriter(OUTPUT_EXCEL, engine="openpyxl") as writer:
    summary_df.to_excel(writer, sheet_name="Summary", index=False)
    metadata_df.to_excel(writer, sheet_name="All_metadata", index=False)

print(f"\nSaved to: {OUTPUT_EXCEL}")