#!/usr/bin/env python3

import os
import re
import pydicom
import pandas as pd

# =====================================================
# KONFIGURASJON
# =====================================================

BASE_FOLDER = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\CH4_Characterization2"

OUTPUT_EXCEL = "CH4_characterisation_metadata.xlsx"

# Hvilken DICOM-fil/slice som skal brukes for metadata
# 0 = første DICOM-fil etter sortering
SLICE_INDEX_TO_USE = 0

SECONDS_PER_DAY = 86400


# =====================================================
# HJELPEFUNKSJONER
# =====================================================

def dicom_sort_key(filename):
    nums = re.findall(r"\d+", filename)
    if nums:
        return int(nums[-1])
    return 10**9


def get_dicom_files_sorted(dicom_folder):
    dicom_files = [
        f for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]
    dicom_files.sort(key=dicom_sort_key)
    return dicom_files


def safe_get(ds, name, default=None):
    value = getattr(ds, name, default)

    if value is None:
        return default

    # Gjør pydicom-verdier trygge for Excel
    try:
        if isinstance(value, (list, tuple)):
            return ", ".join(str(v) for v in value)
        return str(value)
    except Exception:
        return default


def get_scan_time(ds):
    """
    Returnerer scan-tid som:
    - sekunder siden midnatt
    - tekstformat HH:MM:SS
    """

    for tag in ["AcquisitionTime", "SeriesTime", "StudyTime"]:
        t = getattr(ds, tag, None)

        if t:
            t = str(t).split(".")[0]

            if len(t) >= 6:
                hh = int(t[0:2])
                mm = int(t[2:4])
                ss = int(t[4:6])

                return hh * 3600 + mm * 60 + ss, f"{hh:02d}:{mm:02d}:{ss:02d}"

    return None, "Unknown"


def choose_slice(scan_no, number_of_dicoms):
    """
    Brukes hvis noen scans trenger en annen slice.
    Endre her dersom du har spesielle scans.
    """

    # Eksempel:
    # if scan_no == 36:
    #     return 32

    slice_index = SLICE_INDEX_TO_USE

    if slice_index >= number_of_dicoms:
        return None

    return slice_index


def extract_selected_metadata(ds, scan_no, slice_index, dicom_file, number_of_dicoms):
    """
    Henter ut de viktigste metadataene.
    Du kan legge til flere DICOM-tags her senere.
    """

    pixel_spacing = getattr(ds, "PixelSpacing", [None, None])

    if pixel_spacing is not None and len(pixel_spacing) >= 2:
        pixel_spacing_y = pixel_spacing[0]
        pixel_spacing_x = pixel_spacing[1]
    else:
        pixel_spacing_y = None
        pixel_spacing_x = None

    metadata = {
        "Scan": scan_no,
        "Slice_index": slice_index,
        "DICOM_file": os.path.basename(dicom_file),
        "Number_of_DICOM_files": number_of_dicoms,

        # Tid
        "StudyDate": safe_get(ds, "StudyDate"),
        "StudyTime": safe_get(ds, "StudyTime"),
        "SeriesTime": safe_get(ds, "SeriesTime"),
        "AcquisitionTime": safe_get(ds, "AcquisitionTime"),

        # Beskrivelse av scan/protokoll
        "SeriesDescription": safe_get(ds, "SeriesDescription"),
        "ProtocolName": safe_get(ds, "ProtocolName"),
        "SequenceName": safe_get(ds, "SequenceName"),
        "ScanningSequence": safe_get(ds, "ScanningSequence"),
        "SequenceVariant": safe_get(ds, "SequenceVariant"),
        "ScanOptions": safe_get(ds, "ScanOptions"),

        # Viktige scanparametre
        "EchoTime_ms": safe_get(ds, "EchoTime"),
        "RepetitionTime_ms": safe_get(ds, "RepetitionTime"),
        "FlipAngle_deg": safe_get(ds, "FlipAngle"),
        "EchoTrainLength": safe_get(ds, "EchoTrainLength"),
        "NumberOfAverages": safe_get(ds, "NumberOfAverages"),
        "MagneticFieldStrength_T": safe_get(ds, "MagneticFieldStrength"),

        # Geometri
        "Rows": safe_get(ds, "Rows"),
        "Columns": safe_get(ds, "Columns"),
        "PixelSpacingY_mm": pixel_spacing_y,
        "PixelSpacingX_mm": pixel_spacing_x,
        "SliceThickness_mm": safe_get(ds, "SliceThickness"),
        "SpacingBetweenSlices_mm": safe_get(ds, "SpacingBetweenSlices"),
        "ImagePositionPatient": safe_get(ds, "ImagePositionPatient"),
        "ImageOrientationPatient": safe_get(ds, "ImageOrientationPatient"),

        # Pasient/studie info, ofte anonymisert
        "PatientName": safe_get(ds, "PatientName"),
        "PatientID": safe_get(ds, "PatientID"),
        "StudyDescription": safe_get(ds, "StudyDescription"),
        "Manufacturer": safe_get(ds, "Manufacturer"),
        "ManufacturerModelName": safe_get(ds, "ManufacturerModelName"),
        "SoftwareVersions": safe_get(ds, "SoftwareVersions"),
    }

    return metadata


def extract_all_metadata_flat(ds, scan_no, slice_index):
    """
    Lager en lang tabell med alle DICOM-tags.
    Nyttig hvis du vil lete etter hvilke metadata som finnes.
    """

    rows = []

    for elem in ds.iterall():
        if elem.keyword == "PixelData":
            continue

        try:
            value = elem.value

            # Ikke lagre veldig lange verdier
            value_str = str(value)
            if len(value_str) > 500:
                value_str = value_str[:500] + "..."

            rows.append({
                "Scan": scan_no,
                "Slice_index": slice_index,
                "Tag": str(elem.tag),
                "Keyword": elem.keyword,
                "Name": elem.name,
                "VR": elem.VR,
                "Value": value_str
            })

        except Exception:
            continue

    return rows


# =====================================================
# MAIN
# =====================================================

def main():
    selected_metadata_rows = []
    all_metadata_rows = []

    first_time = None
    previous_time = None
    day_offset = 0

    scan_folders = sorted([
        int(f) for f in os.listdir(BASE_FOLDER)
        if f.isdigit()
    ])

    print("Fant scan-mapper:")
    print(scan_folders)

    for scan_no in scan_folders:
        dicom_folder = os.path.join(
            BASE_FOLDER,
            str(scan_no),
            "pdata",
            "1",
            "dicom"
        )

        if not os.path.isdir(dicom_folder):
            print(f"[SCAN {scan_no}] DICOM folder missing")
            continue

        dicom_files = get_dicom_files_sorted(dicom_folder)

        if len(dicom_files) == 0:
            print(f"[SCAN {scan_no}] no DICOM files found")
            continue

        slice_index = choose_slice(scan_no, len(dicom_files))

        if slice_index is None:
            print(f"[SCAN {scan_no}] selected slice does not exist")
            continue

        dicom_file = os.path.join(dicom_folder, dicom_files[slice_index])

        print(f"[SCAN {scan_no}] reading slice {slice_index}: {dicom_files[slice_index]}")

        # stop_before_pixels=True gjør lesing raskere fordi vi bare trenger metadata
        ds = pydicom.dcmread(dicom_file, stop_before_pixels=True)

        metadata = extract_selected_metadata(
            ds=ds,
            scan_no=scan_no,
            slice_index=slice_index,
            dicom_file=dicom_file,
            number_of_dicoms=len(dicom_files)
        )

        # Relativ tid med håndtering av midnatt
        time_sec, time_str = get_scan_time(ds)
        rel_time = None

        if time_sec is not None:
            if previous_time is not None and time_sec < previous_time:
                day_offset += SECONDS_PER_DAY

            absolute_time = time_sec + day_offset

            if first_time is None:
                first_time = absolute_time

            rel_time = absolute_time - first_time
            previous_time = time_sec

        metadata["Time"] = time_str
        metadata["Time_seconds"] = rel_time
        metadata["Time_hours"] = rel_time / 3600 if rel_time is not None else None

        selected_metadata_rows.append(metadata)

        # Lang metadata-tabell med alle tags
        all_metadata_rows.extend(
            extract_all_metadata_flat(ds, scan_no, slice_index)
        )

    df_selected = pd.DataFrame(selected_metadata_rows)
    df_all = pd.DataFrame(all_metadata_rows)

    output_path = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df_selected.to_excel(writer, sheet_name="Selected_Metadata", index=False)
        df_all.to_excel(writer, sheet_name="All_DICOM_Tags", index=False)

    print("\nFerdig!")
    print(f"Lagret metadata til:")
    print(output_path)


if __name__ == "__main__":
    main()