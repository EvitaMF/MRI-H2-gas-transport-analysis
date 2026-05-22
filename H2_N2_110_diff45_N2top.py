import os
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import pandas as pd

# =====================================================
# KONFIGURASJON
# =====================================================

BASE_FOLDER = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\diffusion_study\2026 02 Diffusion\MRI Scan data\H2 N2 110 diff45 N2top"

OUTPUT_EXCEL = "H2_N2_diffusion_N2top_results.xlsx"

SAVE_PNG = True
PNG_DPI = 200

# -----------------------------------------------------
# ROI-INNSTILLINGER
# ROI1 = venstre/top/N2
# ROI2 = høyre/bottom/H2
# -----------------------------------------------------

ROI1_LABEL = "ROI1_top_N2"
ROI2_LABEL = "ROI2_bottom_H2"

RECT1_TOP_MM = 20.0
RECT1_HEIGHT_MM = 70.0
RECT1_CENTER_X_MM = 8.0

RECT2_TOP_MM = 20.0
RECT2_HEIGHT_MM = 70.0
RECT2_CENTER_X_MM = 48.0

# Fast bredde i piksler, så ROI1 og ROI2 alltid blir like brede
ROI_WIDTH_PX = 3

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


def find_middle_dicom(dicom_folder):
    if not os.path.exists(dicom_folder):
        return None, None

    dicom_files = sorted([
        f for f in os.listdir(dicom_folder)
        if f.endswith(".dcm") and f.startswith("MRIm")
    ])

    if len(dicom_files) == 0:
        return None, None

    mid_index = len(dicom_files) // 2
    dicom_file = dicom_files[mid_index]

    try:
        slice_no = int(dicom_file.replace("MRIm", "").replace(".dcm", ""))
    except:
        slice_no = None

    return os.path.join(dicom_folder, dicom_file), slice_no


def rect_bbox_fixed_width_px(shape, pixel_spacing, top_mm, height_mm, width_px, centre_x_mm):
    rows, cols = shape

    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    y0_raw = int(np.floor(top_mm / dy_mm))
    y1_raw = int(np.ceil((top_mm + height_mm) / dy_mm))

    centre_x_px = int(round(centre_x_mm / dx_mm))

    half_w = width_px // 2

    if width_px % 2 == 0:
        x0_raw = centre_x_px - half_w
        x1_raw = centre_x_px + half_w
    else:
        x0_raw = centre_x_px - half_w
        x1_raw = centre_x_px + half_w + 1

    y0 = max(0, y0_raw)
    y1 = min(rows, y1_raw)

    x0 = max(0, x0_raw)
    x1 = min(cols, x1_raw)

    clipped = (
        x0 != x0_raw or
        x1 != x1_raw or
        y0 != y0_raw or
        y1 != y1_raw
    )

    return (x0, y0, x1, y1), (x0_raw, y0_raw, x1_raw, y1_raw), clipped


def normalise_for_display(img):
    img = np.asarray(img, dtype=float)

    denom = img.max() - img.min()

    if denom == 0:
        return np.zeros_like(img)

    return (img - img.min()) / denom


def save_png_with_rects(img, bbox1, bbox2, path, title):
    fig, ax = plt.subplots()

    ax.imshow(normalise_for_display(img), cmap="gray")

    rect1 = plt.Rectangle(
        (bbox1[0], bbox1[1]),
        bbox1[2] - bbox1[0],
        bbox1[3] - bbox1[1],
        fill=False,
        linewidth=2,
        edgecolor="red"
    )

    rect2 = plt.Rectangle(
        (bbox2[0], bbox2[1]),
        bbox2[2] - bbox2[0],
        bbox2[3] - bbox2[1],
        fill=False,
        linewidth=2,
        edgecolor="blue"
    )

    ax.add_patch(rect1)
    ax.add_patch(rect2)

    ax.set_title(title)
    ax.axis("off")

    plt.tight_layout()
    fig.savefig(path, dpi=PNG_DPI)
    plt.close(fig)


def main():
    results = []

    first_time = None
    previous_time = None
    day_offset = 0
    SECONDS_PER_DAY = 86400

    scan_folders = sorted([
        int(f) for f in os.listdir(BASE_FOLDER)
        if f.isdigit()
    ])

    for scan_no in scan_folders:

        dicom_folder = os.path.join(
            BASE_FOLDER,
            str(scan_no),
            "pdata",
            "1",
            "dicom"
        )

        dicom_file, target_slice = find_middle_dicom(dicom_folder)

        if dicom_file is None:
            print(f"[SCAN {scan_no}] no DICOM file found")
            continue

        print(f"\n[SCAN {scan_no}] using middle slice {target_slice} ({os.path.basename(dicom_file)})")

        ds = pydicom.dcmread(dicom_file)

        img = ds.pixel_array.astype(float)

        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        img = img * slope + intercept

        pixel_spacing = getattr(ds, "PixelSpacing", [1, 1])

        bbox1, bbox1_raw, roi1_clipped = rect_bbox_fixed_width_px(
            img.shape,
            pixel_spacing,
            RECT1_TOP_MM,
            RECT1_HEIGHT_MM,
            ROI_WIDTH_PX,
            RECT1_CENTER_X_MM
        )

        bbox2, bbox2_raw, roi2_clipped = rect_bbox_fixed_width_px(
            img.shape,
            pixel_spacing,
            RECT2_TOP_MM,
            RECT2_HEIGHT_MM,
            ROI_WIDTH_PX,
            RECT2_CENTER_X_MM
        )

        roi1_width_px = bbox1[2] - bbox1[0]
        roi1_height_px = bbox1[3] - bbox1[1]

        roi2_width_px = bbox2[2] - bbox2[0]
        roi2_height_px = bbox2[3] - bbox2[1]

        print("image shape:", img.shape)
        print("pixel spacing:", pixel_spacing)
        print("bbox1:", bbox1, "raw:", bbox1_raw,
              "width_px:", roi1_width_px,
              "height_px:", roi1_height_px,
              "clipped:", roi1_clipped)
        print("bbox2:", bbox2, "raw:", bbox2_raw,
              "width_px:", roi2_width_px,
              "height_px:", roi2_height_px,
              "clipped:", roi2_clipped)

        if roi1_width_px != roi2_width_px or roi1_height_px != roi2_height_px:
            print("WARNING: ROI1 and ROI2 are not the same pixel size.")

        if roi1_clipped or roi2_clipped:
            print("WARNING: One ROI has been clipped by the image boundary.")

        roi1 = img[bbox1[1]:bbox1[3], bbox1[0]:bbox1[2]]
        roi2 = img[bbox2[1]:bbox2[3], bbox2[0]:bbox2[2]]

        roi1_mean = float(np.mean(roi1))
        roi1_median = float(np.median(roi1))
        roi1_std = float(np.std(roi1))

        roi2_mean = float(np.mean(roi2))
        roi2_median = float(np.median(roi2))
        roi2_std = float(np.std(roi2))

        time_sec, time_str = get_scan_time(ds)

        if time_sec is not None:
            if previous_time is not None and time_sec < previous_time:
                day_offset += SECONDS_PER_DAY

            absolute_time = time_sec + day_offset

            if first_time is None:
                first_time = absolute_time

            rel_time = absolute_time - first_time
            previous_time = time_sec

        else:
            rel_time = None

        results.append({
            "Scan": scan_no,
            "Slice": target_slice,
            "DICOM_file": os.path.basename(dicom_file),
            "Time": time_str,
            "Time_seconds": rel_time,

            f"{ROI1_LABEL}_mean": roi1_mean,
            f"{ROI1_LABEL}_median": roi1_median,
            f"{ROI1_LABEL}_std": roi1_std,

            f"{ROI2_LABEL}_mean": roi2_mean,
            f"{ROI2_LABEL}_median": roi2_median,
            f"{ROI2_LABEL}_std": roi2_std,

            "ROI1_width_px": roi1_width_px,
            "ROI1_height_px": roi1_height_px,
            "ROI1_clipped": roi1_clipped,

            "ROI2_width_px": roi2_width_px,
            "ROI2_height_px": roi2_height_px,
            "ROI2_clipped": roi2_clipped,

            "EchoTime": getattr(ds, "EchoTime", None),
            "RepetitionTime": getattr(ds, "RepetitionTime", None),
            "FlipAngle": getattr(ds, "FlipAngle", None),

            "PixelSpacingY": pixel_spacing[0],
            "PixelSpacingX": pixel_spacing[1],
            "SliceThickness": getattr(ds, "SliceThickness", None),
            "Rows": getattr(ds, "Rows", None),
            "Columns": getattr(ds, "Columns", None)
        })

        if SAVE_PNG:
            out_png = os.path.join(
                dicom_folder,
                f"scan{scan_no}_ROI.png"
            )

            title = f"Scan {scan_no} – {time_str} – slice {target_slice}"
            save_png_with_rects(img, bbox1, bbox2, out_png, title)

    df = pd.DataFrame(results)
    df["Time_minutes"] = df["Time_seconds"] / 60

    excel_path = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    df.to_excel(excel_path, index=False)

    print("\nExcel lagret:", excel_path)


if __name__ == "__main__":
    main()