#!/usr/bin/env python3
"""
T2_batch_mmROI.py

- Kjører T2-analyse for flere scan i samme CH4_Characterization2-mappe.
- Hvert scan ligger i: BASE_FOLDER/<scan_no>/pdata/1/dicom
- ROI: én eller flere sirkler definert i millimeter (CIRCLE_CENTERS_MM, CIRCLE_RADIUS_MM)
- For hver scan:
    * Leser DICOM-serie og sorterer på EchoTime
    * Lager ROI-mask fra mm -> pixler via PixelSpacing
    * Beregner gjennomsnittssignal i ROI for hver TE
    * Fitter T2 (S(TE) = S0 * exp(-TE/T2)) og beregner R²
    * Lager PNG med ROI på første ekko

- Etter alle scan:
    * Lager ETT Excel-dokument med tre sheets:
        - "T2_Results": én rad per scan
        - "DataPoints": TE_ms nedover, én kolonne per scan (mean signal)
        - "Metadata": én rad per scan med de metadata-feltene dere ønsker
"""

import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =====================================================
# KONFIGURASJON – ENDRE HER
# =====================================================

BASE_FOLDER = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\CH4_Characterization2"

# Hvilke scan-mapper vil du prosessere (under BASE_FOLDER)?
SCAN_NUMBERS = [7, 17, 26, 35, 44, 53, 62, 71, 80, 89, 98]  # f.eks. [1, 3, 7, 10]

# ROI i mm (samme for alle scan)
CIRCLE_CENTERS_MM = [(29.5, 29.75)]  # liste av (x_mm, y_mm)
CIRCLE_RADIUS_MM = 13.0             # radius i mm

# Output Excel-fil (samlet for alle scan)
OUTPUT_EXCEL = "T2_all_scans_results.xlsx"

# =====================================================


def exp_decay(te, s0, t2):
    """Eksponensiell T2-funksjon."""
    return s0 * np.exp(-te / t2)


def load_dicom_series(dicom_folder):
    """Leser og sorterer DICOM-serie på EchoTime (fallback InstanceNumber)."""
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


def create_circle_mask_mm(shape, pixel_spacing, centers_mm, radius_mm):
    """
    Lager én samlet ROI-mask for en eller flere sirkler gitt i mm.
    pixel_spacing: [dy_mm, dx_mm]
    centers_mm: liste av (x_mm, y_mm)
    """
    rows, cols = shape
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    Y, X = np.ogrid[:rows, :cols]
    mask_total = np.zeros(shape, dtype=bool)

    for (cx_mm, cy_mm) in centers_mm:
        cx_p = cx_mm / dx_mm
        cy_p = cy_mm / dy_mm
        radius_p = radius_mm / dx_mm  # antar kvadratiske pixler
        dist2 = (X - cx_p) ** 2 + (Y - cy_p) ** 2
        mask_total |= (dist2 <= radius_p ** 2)

    return mask_total


def fit_t2_and_r2(te_list, signal_list):
    """
    Fitter T2 og beregner R^2.
    Returnerer: s0, t2, s0_err, t2_err, r2
    """
    te = np.array(te_list, dtype=float)
    sig = np.array(signal_list, dtype=float)

    valid = np.isfinite(te) & np.isfinite(sig) & (sig > 0)
    te = te[valid]
    sig = sig[valid]

    if len(te) < 3:
        raise RuntimeError("For få punkt til å fitte T2!")

    s0_init = sig.max()
    t2_init = (te.max() - te.min()) / 2 if te.max() > te.min() else 100.0

    popt, pcov = curve_fit(exp_decay, te, sig, p0=[s0_init, t2_init], maxfev=10000)
    s0, t2 = popt
    s0_err, t2_err = np.sqrt(np.diag(pcov))

    # R^2
    fit_vals = exp_decay(te, s0, t2)
    residuals = sig - fit_vals
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return s0, t2, s0_err, t2_err, r2


def main():
    # Samlere for Excel
    t2_results_rows = []      # til T2_Results (én rad per scan)
    metadata_rows = []        # til Metadata (én rad per scan)
    scan_signal_map = {}      # til DataPoints: {scan_no: {TE: mean_signal}}
    all_te_values = set()     # alle TE_ms som dukker opp i noen scan

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(
            BASE_FOLDER, str(scan_no), "pdata", "1", "dicom"
        )
        print(f"\n[SCAN {scan_no}] Mappe: {dicom_folder}")

        images, echo_times, ds0 = load_dicom_series(dicom_folder)
        img0 = images[0]
        rows, cols = img0.shape

        pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])
        print(f"[SCAN {scan_no}] PixelSpacing: {pixel_spacing}")

        # ROI-mask
        mask = create_circle_mask_mm(
            shape=img0.shape,
            pixel_spacing=pixel_spacing,
            centers_mm=CIRCLE_CENTERS_MM,
            radius_mm=CIRCLE_RADIUS_MM,
        )

        roi_area_px = int(mask.sum())
        roi_fraction = roi_area_px / (rows * cols) * 100.0
        print(f"[SCAN {scan_no}] ROI pixler: {roi_area_px}, {roi_fraction:.2f}% av bildet")

        # Mean signal per TE
        mean_signals = []
        te_to_sig = {}
        for te, img in zip(echo_times, images):
            val = float(np.mean(img[mask]))
            mean_signals.append(val)
            te_to_sig[float(te)] = val
            all_te_values.add(float(te))
            print(f"[SCAN {scan_no}] TE={te} ms, mean_signal={val:.2f}")

        scan_signal_map[scan_no] = te_to_sig

        # Fit T2
        try:
            s0, t2, s0_err, t2_err, r2 = fit_t2_and_r2(echo_times, mean_signals)
            print(f"[SCAN {scan_no}] T2 = {t2:.2f} ± {t2_err:.2f} ms, R² = {r2:.4f}")
        except Exception as e:
            print(f"[SCAN {scan_no}] ADVARSEL: klarte ikke å fitte T2: {e}")
            s0 = t2 = s0_err = t2_err = r2 = np.nan

        # PNG med ROI
        fig, ax = plt.subplots()
        ax.imshow(img0, cmap="gray")

        dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
        for (cx_mm, cy_mm) in CIRCLE_CENTERS_MM:
            cx_p = cx_mm / dx_mm
            cy_p = cy_mm / dy_mm
            radius_p = CIRCLE_RADIUS_MM / dx_mm
            circ = plt.Circle((cx_p, cy_p), radius_p, fill=False, linewidth=2)
            ax.add_patch(circ)

        ax.set_title(f"Scan {scan_no} – ROI")
        ax.axis("off")
        png_name = f"T2_mmCircleROI_scan{scan_no}.png"
        out_png = os.path.join(dicom_folder, png_name)
        plt.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[SCAN {scan_no}] Lagret PNG: {out_png}")

        # ---------- Bygg rader til Excel ----------

        # 1) T2_Results
        t2_results_rows.append(
            {
                "Scan_no": scan_no,
                "T2_ms": t2,
                "T2_error_ms": t2_err,
                "R2": r2,
            }
        )

        # 2) Metadata – nøyaktig kolonner du oppga
        # TE_list_ms blir en kommaseparert streng
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

    # =====================================================
    # Etter løkken: bygg DataFrames og skriv til Excel
    # =====================================================

    # T2_Results-sheet
    df_t2 = pd.DataFrame(t2_results_rows, columns=["Scan_no", "T2_ms", "T2_error_ms", "R2"])

    # Metadata-sheet – behold kolonnerekkefølgen
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

    # DataPoints-sheet – TE_ms nedover, én kolonne per scan
    all_te_sorted = sorted(all_te_values)
    dp_data = {"TE_ms": all_te_sorted}
    for scan_no, te_to_sig in scan_signal_map.items():
        col_name = f"Scan_{scan_no}"
        dp_data[col_name] = [te_to_sig.get(te, np.nan) for te in all_te_sorted]

    df_dp = pd.DataFrame(dp_data)

    # Skriv til én Excel-fil
    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_t2.to_excel(writer, index=False, sheet_name="T2_Results")
        df_dp.to_excel(writer, index=False, sheet_name="DataPoints")
        df_meta.to_excel(writer, index=False, sheet_name="Metadata")

    print(f"\n[OUTPUT] Lagret samlet Excel-fil: {out_xlsx}")


if __name__ == "__main__":
    main()
