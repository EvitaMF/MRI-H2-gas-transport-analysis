#!/usr/bin/env python3

import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Rectangle

# =====================================================
# KONFIGURASJON
# =====================================================

BASE_FOLDER = r"/Users/camillaomland/Library/CloudStorage/OneDrive-UniversityofBergen/MRI_H2/2026_02_Diffusion/MRIScanData/2026_05_CH4_N2/MRIscandata/CH4_N2_120_diff45V2"

OUTPUT_EXCEL = "CH4_N2_120_diff45V2_diffusion_results.xlsx"

# -----------------------------------------------------
# Scan/slice-valg
# -----------------------------------------------------

PRE_FLIP_SCAN_MIN = 8
PRE_FLIP_SCAN_MAX = 127
PRE_FLIP_SLICE = 7

POST_FLIP_SCAN_MIN = 128
POST_FLIP_SCAN_MAX = 190
POST_FLIP_SLICE = 7

SWAP_ROIS_AFTER_FLIP = True

# -----------------------------------------------------
# PNG-lagring
# -----------------------------------------------------

SAVE_SELECTED_PNGS = True
SHOW_SELECTED_PNGS = False

PNG_DPI = 200

FIRST_SCAN = 8
LAST_PRE_FLIP_SCAN = 127
FIRST_POST_FLIP_SCAN = 128
LAST_POST_FLIP_SCAN = 190

ROI_FIRST_SCAN_PNG_NAME = "ROI_scan008_first_after_valve_opening_CH4_N2_120_45V2.png"
ROI_LAST_PRE_FLIP_PNG_NAME = "ROI_scan127_last_pre_flip_CH4_N2_120_45V2.png"
ROI_FIRST_POST_FLIP_PNG_NAME = "ROI_scan128_first_post_flip_CH4_N2_120_45V2.png"
ROI_LAST_POST_FLIP_PNG_NAME = "ROI_scan190_last_post_flip_CH4_N2_120_45V2.png"

# -----------------------------------------------------
# Farger
# -----------------------------------------------------

COLOR_CH4 = "tab:red"
COLOR_N2 = "tab:blue"

# =====================================================
# ROI-parametere FØR flip
# =====================================================

RECT1_PRE_TOP_MM = 13.0
RECT1_PRE_HEIGHT_MM = 89.0
RECT1_PRE_WIDTH_MM = 8.0
RECT1_PRE_CENTER_X_MM = 12.7

RECT2_PRE_TOP_MM = 13.0
RECT2_PRE_HEIGHT_MM = 89.0
RECT2_PRE_WIDTH_MM = 8.0
RECT2_PRE_CENTER_X_MM = 52.3

# =====================================================
# ROI-parametere ETTER flip
# =====================================================

RECT1_POST_TOP_MM = 13.0
RECT1_POST_HEIGHT_MM = 89.0
RECT1_POST_WIDTH_MM = 8.0
RECT1_POST_CENTER_X_MM = 14.5

RECT2_POST_TOP_MM = 13.0
RECT2_POST_HEIGHT_MM = 89.0
RECT2_POST_WIDTH_MM = 8.0
RECT2_POST_CENTER_X_MM = 55.0


# =====================================================
# HJELPEFUNKSJONER
# =====================================================

def get_scan_time(ds):
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


def choose_slice(scan_no):
    if PRE_FLIP_SCAN_MIN <= scan_no <= PRE_FLIP_SCAN_MAX:
        return PRE_FLIP_SLICE

    if POST_FLIP_SCAN_MIN <= scan_no <= POST_FLIP_SCAN_MAX:
        return POST_FLIP_SLICE

    return None


def is_post_flip(scan_no):
    return POST_FLIP_SCAN_MIN <= scan_no <= POST_FLIP_SCAN_MAX


def rect_bbox_from_mm(shape, pixel_spacing, top_mm, height_mm, width_mm, centre_x_mm):
    rows, cols = shape

    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    y0 = int(np.floor(top_mm / dy_mm))
    y1 = int(np.ceil((top_mm + height_mm) / dy_mm))

    centre_x_px = centre_x_mm / dx_mm
    half_w_px = (width_mm / dx_mm) / 2.0

    x0 = int(np.floor(centre_x_px - half_w_px))
    x1 = int(np.ceil(centre_x_px + half_w_px))

    y0 = max(0, y0)
    y1 = min(rows, y1)

    x0 = max(0, x0)
    x1 = min(cols, x1)

    return (x0, y0, x1, y1)


def normalise_for_display(img):
    img = np.asarray(img, dtype=float)

    vmin = np.nanpercentile(img, 1)
    vmax = np.nanpercentile(img, 99.5)

    if vmax <= vmin:
        return np.zeros_like(img)

    img_clip = np.clip(img, vmin, vmax)
    return (img_clip - vmin) / (vmax - vmin)


def safe_stats(values):
    values = np.asarray(values, dtype=float).ravel()

    if values.size == 0:
        return np.nan, np.nan, np.nan, 0

    return (
        float(np.mean(values)),
        float(np.median(values)),
        float(np.std(values)),
        int(values.size)
    )


def add_roi_label(ax, x, y, label, color):
    ax.text(
        x,
        y,
        label,
        color=color,
        fontsize=9,
        fontweight="bold",
        ha="left",
        va="bottom",
        rotation=90,
        rotation_mode="anchor"
    )


def add_rect_roi(ax, bbox, color, label):
    x0, y0, x1, y1 = bbox

    patch = Rectangle(
        (x0, y0),
        x1 - x0,
        y1 - y0,
        fill=False,
        linewidth=2,
        edgecolor=color
    )

    ax.add_patch(patch)

    add_roi_label(
        ax,
        x1 - 2,
        y1 + 5,
        label,
        color
    )


def save_png_with_rois(img, bbox_ch4, bbox_n2, path, title):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(normalise_for_display(img), cmap="gray")

    add_rect_roi(ax, bbox_ch4, COLOR_CH4, "CH$_4$")
    add_rect_roi(ax, bbox_n2, COLOR_N2, "N$_2$")

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=PNG_DPI, bbox_inches="tight")
    plt.close(fig)


def show_png_with_rois(img, bbox_ch4, bbox_n2, title):
    fig, ax = plt.subplots(figsize=(5, 5))

    ax.imshow(normalise_for_display(img), cmap="gray")

    add_rect_roi(ax, bbox_ch4, COLOR_CH4, "CH$_4$")
    add_rect_roi(ax, bbox_n2, COLOR_N2, "N$_2$")

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    plt.show()


def get_roi_bboxes(img_shape, pixel_spacing, scan_no):
    if is_post_flip(scan_no):
        bbox1 = rect_bbox_from_mm(
            img_shape,
            pixel_spacing,
            RECT1_POST_TOP_MM,
            RECT1_POST_HEIGHT_MM,
            RECT1_POST_WIDTH_MM,
            RECT1_POST_CENTER_X_MM
        )

        bbox2 = rect_bbox_from_mm(
            img_shape,
            pixel_spacing,
            RECT2_POST_TOP_MM,
            RECT2_POST_HEIGHT_MM,
            RECT2_POST_WIDTH_MM,
            RECT2_POST_CENTER_X_MM
        )

    else:
        bbox1 = rect_bbox_from_mm(
            img_shape,
            pixel_spacing,
            RECT1_PRE_TOP_MM,
            RECT1_PRE_HEIGHT_MM,
            RECT1_PRE_WIDTH_MM,
            RECT1_PRE_CENTER_X_MM
        )

        bbox2 = rect_bbox_from_mm(
            img_shape,
            pixel_spacing,
            RECT2_PRE_TOP_MM,
            RECT2_PRE_HEIGHT_MM,
            RECT2_PRE_WIDTH_MM,
            RECT2_PRE_CENTER_X_MM
        )

    return bbox1, bbox2


def extract_roi_values(img, bbox):
    x0, y0, x1, y1 = bbox
    return img[y0:y1, x0:x1]


# =====================================================
# MAIN
# =====================================================

def main():
    results = []

    first_time = None
    previous_time = None
    day_offset = 0
    SECONDS_PER_DAY = 86400

    script_dir = os.path.dirname(os.path.abspath(__file__))

    roi_first_scan_png_path = os.path.join(script_dir, ROI_FIRST_SCAN_PNG_NAME)
    roi_last_pre_flip_png_path = os.path.join(script_dir, ROI_LAST_PRE_FLIP_PNG_NAME)
    roi_first_post_flip_png_path = os.path.join(script_dir, ROI_FIRST_POST_FLIP_PNG_NAME)
    roi_last_post_flip_png_path = os.path.join(script_dir, ROI_LAST_POST_FLIP_PNG_NAME)

    if not os.path.exists(BASE_FOLDER):
        raise FileNotFoundError(f"BASE_FOLDER finnes ikke:\n{BASE_FOLDER}")

    scan_folders = sorted(
        [
            int(f)
            for f in os.listdir(BASE_FOLDER)
            if f.isdigit()
        ]
    )

    print("Fant scan-mapper:", scan_folders)

    for scan_no in scan_folders:
        target_slice = choose_slice(scan_no)

        if target_slice is None:
            print(f"[SCAN {scan_no}] skipped")
            continue

        dicom_folder = os.path.join(
            BASE_FOLDER,
            str(scan_no),
            "pdata",
            "1",
            "dicom"
        )

        dicom_file = os.path.join(
            dicom_folder,
            f"MRIm{target_slice:02d}.dcm"
        )

        if not os.path.exists(dicom_file):
            print(f"[SCAN {scan_no}] file missing: {dicom_file}")
            continue

        print(f"[SCAN {scan_no}] using slice {target_slice}")

        ds = pydicom.dcmread(dicom_file)

        img = ds.pixel_array.astype(float)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))

        img = img * slope + intercept

        pixel_spacing = getattr(ds, "PixelSpacing", [1, 1])

        bbox1, bbox2 = get_roi_bboxes(img.shape, pixel_spacing, scan_no)

        roi1_values = extract_roi_values(img, bbox1)
        roi2_values = extract_roi_values(img, bbox2)

        if SWAP_ROIS_AFTER_FLIP and is_post_flip(scan_no):
            roi1_values, roi2_values = roi2_values, roi1_values
            bbox_ch4, bbox_n2 = bbox2, bbox1
        else:
            bbox_ch4, bbox_n2 = bbox1, bbox2

        roi1_mean, roi1_median, roi1_std, roi1_n_pixels = safe_stats(roi1_values)
        roi2_mean, roi2_median, roi2_std, roi2_n_pixels = safe_stats(roi2_values)

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

        results.append({
            "Scan": scan_no,
            "Slice": target_slice,
            "Post_flip": is_post_flip(scan_no),
            "Time": time_str,
            "Time_seconds": rel_time,

            "ROI1_gas": "CH4",
            "ROI2_gas": "N2",

            "ROI1_mean": roi1_mean,
            "ROI1_median": roi1_median,
            "ROI1_std": roi1_std,
            "ROI1_n_pixels": roi1_n_pixels,

            "ROI2_mean": roi2_mean,
            "ROI2_median": roi2_median,
            "ROI2_std": roi2_std,
            "ROI2_n_pixels": roi2_n_pixels,

            "EchoTime": getattr(ds, "EchoTime", None),
            "RepetitionTime": getattr(ds, "RepetitionTime", None),
            "FlipAngle": getattr(ds, "FlipAngle", None),

            "PixelSpacingY": pixel_spacing[0],
            "PixelSpacingX": pixel_spacing[1],
            "SliceThickness": getattr(ds, "SliceThickness", None),
            "Rows": getattr(ds, "Rows", None),
            "Columns": getattr(ds, "Columns", None)
        })

        if SAVE_SELECTED_PNGS and scan_no == FIRST_SCAN:
            title = f"First scan after valve opening | Scan {scan_no} | Slice {target_slice} | {time_str}"
            save_png_with_rois(img, bbox_ch4, bbox_n2, roi_first_scan_png_path, title)
            print(f"ROI-PNG lagret: {roi_first_scan_png_path}")

            if SHOW_SELECTED_PNGS:
                show_png_with_rois(img, bbox_ch4, bbox_n2, title)

        if SAVE_SELECTED_PNGS and scan_no == LAST_PRE_FLIP_SCAN:
            title = f"Last scan before flipping | Scan {scan_no} | Slice {target_slice} | {time_str}"
            save_png_with_rois(img, bbox_ch4, bbox_n2, roi_last_pre_flip_png_path, title)
            print(f"ROI-PNG lagret: {roi_last_pre_flip_png_path}")

            if SHOW_SELECTED_PNGS:
                show_png_with_rois(img, bbox_ch4, bbox_n2, title)

        if SAVE_SELECTED_PNGS and scan_no == FIRST_POST_FLIP_SCAN:
            title = f"First scan after flipping | Scan {scan_no} | Slice {target_slice} | {time_str}"
            save_png_with_rois(img, bbox_ch4, bbox_n2, roi_first_post_flip_png_path, title)
            print(f"ROI-PNG lagret: {roi_first_post_flip_png_path}")

            if SHOW_SELECTED_PNGS:
                show_png_with_rois(img, bbox_ch4, bbox_n2, title)

        if SAVE_SELECTED_PNGS and scan_no == LAST_POST_FLIP_SCAN:
            title = f"Last scan after flipping | Scan {scan_no} | Slice {target_slice} | {time_str}"
            save_png_with_rois(img, bbox_ch4, bbox_n2, roi_last_post_flip_png_path, title)
            print(f"ROI-PNG lagret: {roi_last_post_flip_png_path}")

            if SHOW_SELECTED_PNGS:
                show_png_with_rois(img, bbox_ch4, bbox_n2, title)

    df = pd.DataFrame(results)

    if df.empty:
        print("Ingen resultater ble samlet inn.")
        return

    df["Time_minutes"] = df["Time_seconds"] / 60
    df["Time_hours"] = df["Time_seconds"] / 3600

    excel_path = os.path.join(script_dir, OUTPUT_EXCEL)
    df.to_excel(excel_path, index=False)

    print("\nExcel lagret:", excel_path)


if __name__ == "__main__":
    main()