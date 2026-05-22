#!/usr/bin/env python3
"""
T2_batch_mmRectROI_sagittal.py

- Kjører T2-analyse for sagittale målinger i samme mappe-struktur som axial-scriptet.
- Hvert scan ligger i: BASE_FOLDER/<scan_no>/pdata/1/dicom
- ROI: rektangel definert i millimeter fra bildekoordinater.

ROI-definisjon:
    RECT_TOP_MM      : avstand fra toppen av bildet til øvre kant av ROI [mm]
    RECT_HEIGHT_MM   : høyde på ROI [mm]
    RECT_WIDTH_MM    : bredde på ROI [mm], None = full bildebredde
    RECT_CENTER_X_MM : x-posisjon til senter av ROI [mm], None = midt i bildet

For hver scan:
    * Leser DICOM-serie og sorterer på EchoTime (TE)
    * Lager rektangulær ROI-mask fra mm -> pixler via PixelSpacing
    * Beregner gjennomsnittssignal i ROI for hver TE
    * Fitter T2 på råsignalet: S(TE) = S0 * exp(-TE/T2)
    * Beregner R²
    * Lager PNG med ROI på første ekko

Etter alle scan:
    * Lager ett Excel-dokument med sheets:
        - T2_Results
        - DataPoints
        - Metadata
"""

import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.optimize import curve_fit

# =====================================================
# KONFIGURASJON – ENDRE HER
# =====================================================

BASE_FOLDER = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\CH4_Characterization2"

# Sagittale T2-scan: fyll inn riktige scan-numre her
SCAN_NUMBERS = [8, 18, 27, 36, 45, 54, 63, 72, 81, 90, 99]  # f.eks. [8, 18, 27, 36]

# Rektangulær ROI i mm
RECT_TOP_MM = 15        # mm fra toppen av bildet
RECT_HEIGHT_MM = 70     # mm høyde
RECT_WIDTH_MM = 20       # mm bredde, None = full bredde
RECT_CENTER_X_MM = None # None = midt i bildet

# Output Excel-fil
OUTPUT_EXCEL = "T2_sagittal_rectROI_results.xlsx"

# =====================================================


def exp_decay(te, s0, t2):
    """Eksponensiell T2-funksjon."""
    return s0 * np.exp(-te / t2)


def load_dicom_series(dicom_folder):
    """Leser og sorterer DICOM-serie på EchoTime, med InstanceNumber som fallback."""
    files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]
    if not files:
        raise FileNotFoundError(f"Ingen DICOM-filer i {dicom_folder}")

    datasets = [pydicom.dcmread(f) for f in files]

    def sort_key(ds):
        te = getattr(ds, "EchoTime", None)
        if te is None:
            return getattr(ds, "InstanceNumber", 0)
        return te

    datasets.sort(key=sort_key)

    images = []
    echo_times = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        images.append(arr)

        te_val = getattr(ds, "EchoTime", np.nan)
        echo_times.append(float(te_val) if te_val is not None else np.nan)

    return images, echo_times, datasets[0]


def get_pixel_spacing(ds):
    """Henter PixelSpacing, med ImagerPixelSpacing som fallback."""
    spacing = getattr(ds, "PixelSpacing", None)
    if spacing is None:
        spacing = getattr(ds, "ImagerPixelSpacing", [1.0, 1.0])
    return spacing


def rect_bounds_mm_to_px(shape, pixel_spacing, top_mm, height_mm, width_mm=None, center_x_mm=None):
    """
    Konverterer rektangel gitt i mm til pixelgrenser.

    DICOM PixelSpacing tolkes som [dy_mm, dx_mm].
    y=0 er toppen av bildet, x=0 er venstre kant.
    """
    rows, cols = shape
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    image_width_mm = cols * dx_mm

    y0 = int(round(top_mm / dy_mm))
    y1 = int(round((top_mm + height_mm) / dy_mm))

    if width_mm is None:
        x0 = 0
        x1 = cols
    else:
        if center_x_mm is None:
            center_x_mm = image_width_mm / 2.0
        x0 = int(round((center_x_mm - width_mm / 2.0) / dx_mm))
        x1 = int(round((center_x_mm + width_mm / 2.0) / dx_mm))

    # Klipp til bildekantene, slik at ROI ikke går utenfor bildet
    x0 = max(0, min(cols, x0))
    x1 = max(0, min(cols, x1))
    y0 = max(0, min(rows, y0))
    y1 = max(0, min(rows, y1))

    if x1 <= x0 or y1 <= y0:
        raise ValueError(
            "ROI ble tom etter konvertering til pixler. "
            "Sjekk RECT_TOP_MM, RECT_HEIGHT_MM, RECT_WIDTH_MM og RECT_CENTER_X_MM."
        )

    return x0, x1, y0, y1


def create_rect_mask_mm(shape, pixel_spacing, top_mm, height_mm, width_mm=None, center_x_mm=None):
    """Lager rektangulær ROI-mask fra mm-koordinater."""
    x0, x1, y0, y1 = rect_bounds_mm_to_px(
        shape=shape,
        pixel_spacing=pixel_spacing,
        top_mm=top_mm,
        height_mm=height_mm,
        width_mm=width_mm,
        center_x_mm=center_x_mm,
    )

    mask = np.zeros(shape, dtype=bool)
    mask[y0:y1, x0:x1] = True
    return mask, (x0, x1, y0, y1)


def fit_t2_and_r2(te_list, signal_list):
    """Fitter T2 og beregner R². Returnerer s0, t2, s0_err, t2_err, r2."""
    te = np.array(te_list, dtype=float)
    sig = np.array(signal_list, dtype=float)

    valid = np.isfinite(te) & np.isfinite(sig) & (sig > 0)
    te = te[valid]
    sig = sig[valid]

    if len(te) < 3:
        raise RuntimeError("For få punkt til å fitte T2!")

    s0_init = sig.max()
    t2_init = (te.max() - te.min()) / 2.0 if te.max() > te.min() else 100.0

    popt, pcov = curve_fit(exp_decay, te, sig, p0=[s0_init, t2_init], maxfev=10000)
    s0, t2 = popt
    s0_err, t2_err = np.sqrt(np.diag(pcov))

    fit_vals = exp_decay(te, s0, t2)
    residuals = sig - fit_vals
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return s0, t2, s0_err, t2_err, r2


def main():
    if not SCAN_NUMBERS:
        raise ValueError("SCAN_NUMBERS er tom. Legg inn sagittale T2-scan før du kjører scriptet.")

    t2_results_rows = []
    metadata_rows = []
    scan_signal_map = {}
    all_te_values = set()

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(BASE_FOLDER, str(scan_no), "pdata", "1", "dicom")
        print(f"\n[SCAN {scan_no}] Mappe: {dicom_folder}")

        images, echo_times, ds0 = load_dicom_series(dicom_folder)
        img0 = images[0]
        rows, cols = img0.shape

        pixel_spacing = get_pixel_spacing(ds0)
        print(f"[SCAN {scan_no}] PixelSpacing: {pixel_spacing}")

        mask, rect_px = create_rect_mask_mm(
            shape=img0.shape,
            pixel_spacing=pixel_spacing,
            top_mm=RECT_TOP_MM,
            height_mm=RECT_HEIGHT_MM,
            width_mm=RECT_WIDTH_MM,
            center_x_mm=RECT_CENTER_X_MM,
        )
        x0, x1, y0, y1 = rect_px

        roi_area_px = int(mask.sum())
        roi_fraction = roi_area_px / (rows * cols) * 100.0
        print(
            f"[SCAN {scan_no}] ROI pixler: {roi_area_px}, {roi_fraction:.2f}% av bildet "
            f"(x={x0}:{x1}, y={y0}:{y1})"
        )

        mean_signals = []
        te_to_sig = {}
        for te, img in zip(echo_times, images):
            val = float(np.mean(img[mask]))
            mean_signals.append(val)
            te_to_sig[float(te)] = val
            all_te_values.add(float(te))
            print(f"[SCAN {scan_no}] TE={te} ms, mean_signal={val:.2f}")

        scan_signal_map[scan_no] = te_to_sig

        try:
            s0, t2, s0_err, t2_err, r2 = fit_t2_and_r2(echo_times, mean_signals)
            print(f"[SCAN {scan_no}] T2 = {t2:.2f} ± {t2_err:.2f} ms, R² = {r2:.4f}")
        except Exception as e:
            print(f"[SCAN {scan_no}] ADVARSEL: klarte ikke å fitte T2: {e}")
            s0 = t2 = s0_err = t2_err = r2 = np.nan

        # ---------- PNG med ROI ----------
        fig, ax = plt.subplots()
        ax.imshow(img0, cmap="gray")
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.set_title(f"Scan {scan_no} – sagittal T2 ROI")
        ax.axis("off")

        png_name = f"T2_sagittal_mmRectROI_scan{scan_no}.png"
        out_png = os.path.join(dicom_folder, png_name)
        plt.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[SCAN {scan_no}] Lagret ROI-PNG: {out_png}")

        t2_results_rows.append(
            {
                "Scan_no": scan_no,
                "T2_ms": t2,
                "T2_error_ms": t2_err,
                "R2": r2,
                "ROI_top_mm": RECT_TOP_MM,
                "ROI_height_mm": RECT_HEIGHT_MM,
                "ROI_width_mm": RECT_WIDTH_MM,
                "ROI_center_x_mm": RECT_CENTER_X_MM,
                "ROI_x0_px": x0,
                "ROI_x1_px": x1,
                "ROI_y0_px": y0,
                "ROI_y1_px": y1,
                "ROI_area_px": roi_area_px,
            }
        )

        valid_te = [float(te) for te in echo_times if np.isfinite(te)]
        te_list_str = ",".join(str(te) for te in valid_te)

        metadata_rows.append(
            {
                "Scan no": scan_no,
                "RepetitionTime": getattr(ds0, "RepetitionTime", np.nan),
                "EchoTime": getattr(ds0, "EchoTime", np.nan),
                "FlipAngle": getattr(ds0, "FlipAngle", np.nan),
                "ImagingFrequency": getattr(ds0, "ImagingFrequency", np.nan),
                "NumberOfAverages": getattr(ds0, "NumberOfAverages", np.nan),
                "PixelSpacing": str(pixel_spacing),
                "SliceThickness": getattr(ds0, "SliceThickness", np.nan),
                "SpacingBetweenSlices": getattr(ds0, "SpacingBetweenSlices", np.nan),
                "Manufacturer": getattr(ds0, "Manufacturer", ""),
                "SeriesDescription": getattr(ds0, "SeriesDescription", ""),
                "ProtocolName": getattr(ds0, "ProtocolName", ""),
                "MagneticFieldStrength": getattr(ds0, "MagneticFieldStrength", np.nan),
                "TE_list_ms": te_list_str,
            }
        )

    df_t2 = pd.DataFrame(t2_results_rows)

    meta_columns = [
        "Scan no",
        "RepetitionTime",
        "EchoTime",
        "FlipAngle",
        "ImagingFrequency",
        "NumberOfAverages",
        "PixelSpacing",
        "SliceThickness",
        "SpacingBetweenSlices",
        "Manufacturer",
        "SeriesDescription",
        "ProtocolName",
        "MagneticFieldStrength",
        "TE_list_ms",
    ]
    df_meta = pd.DataFrame(metadata_rows, columns=meta_columns)

    all_te_sorted = sorted(all_te_values)
    dp_data = {"TE_ms": all_te_sorted}
    for scan_no, te_to_sig in scan_signal_map.items():
        dp_data[f"Scan_{scan_no}"] = [te_to_sig.get(te, np.nan) for te in all_te_sorted]

    df_dp = pd.DataFrame(dp_data)

    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_t2.to_excel(writer, index=False, sheet_name="T2_Results")
        df_dp.to_excel(writer, index=False, sheet_name="DataPoints")
        df_meta.to_excel(writer, index=False, sheet_name="Metadata")

    print(f"\n[OUTPUT] Lagret samlet T2-Excel-fil: {out_xlsx}")


if __name__ == "__main__":
    main()
