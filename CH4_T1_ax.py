#!/usr/bin/env python3
"""
T1_batch_mmROI.py

- Kjører T1-analyse for flere scan i samme mappe-struktur som T2-scriptet.
- Hvert scan ligger i: BASE_FOLDER/<scan_no>/pdata/1/dicom
- ROI: én eller flere sirkler definert i millimeter (CIRCLE_CENTERS_MM, CIRCLE_RADIUS_MM)

For hver scan:
    * Leser DICOM-serie og sorterer på RepetitionTime (TR)
    * Lager ROI-mask fra mm -> pixler via PixelSpacing
    * Beregner gjennomsnittssignal i ROI for hver TR
    * Fitter T1 (S(TR) = S0 * (1 - exp(-TR/T1))) og beregner R²
    * Lager PNG med ROI på første bilde
    * Lager PNG med T1-plot: normaliserte datapunkter + normalisert fit

Etter alle scan:
    * Lager ETT Excel-dokument med tre sheets:
        - "T1_Results": én rad per scan (T1, feil, R²)
        - "DataPoints": TR_ms nedover, råsignal per scan
        - "DataPoints_norm": TR_ms nedover, normalisert signal per scan
        - "Metadata": én rad per scan med DICOM-metadata
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
SCAN_NUMBERS = [9, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100, 105, 106, 107]

# ROI i mm (samme for alle scan)
CIRCLE_CENTERS_MM = [(29.5, 29.75)]  # liste av (x_mm, y_mm)
CIRCLE_RADIUS_MM = 13.0             # radius i mm

# Output Excel-fil (samlet for alle scan)
OUTPUT_EXCEL = "T1_all_scans_results.xlsx"

# =====================================================


def t1_model(tr, s0, t1):
    """T1-funksjon: S(TR) = S0 * (1 - exp(-TR/T1))."""
    return s0 * (1.0 - np.exp(-tr / t1))


def load_dicom_series(dicom_folder):
    """
    Leser og sorterer DICOM-serie på RepetitionTime (TR) (fallback InstanceNumber).
    Returnerer:
        images            : liste av 2D-arrays
        repetition_times  : liste av TR-verdier (ms)
        ds0               : første DICOM-dataset (for metadata)
    """
    files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]
    if not files:
        raise FileNotFoundError(f"Ingen DICOM-filer i {dicom_folder}")

    datasets = [pydicom.dcmread(f) for f in files]

    def sort_key(ds):
        tr = getattr(ds, "RepetitionTime", None)
        if tr is None:
            return getattr(ds, "InstanceNumber", 0)
        return tr

    datasets.sort(key=sort_key)

    images = []
    repetition_times = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)

        # rescale (om tags finnes)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        images.append(arr)

        tr_val = getattr(ds, "RepetitionTime", np.nan)
        repetition_times.append(float(tr_val) if tr_val is not None else np.nan)

    return images, repetition_times, datasets[0]


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


def fit_t1_and_r2(tr_list, signal_list):
    """
    Fitter T1 og beregner R^2.
    Returnerer: s0, t1, s0_err, t1_err, r2
    """
    tr = np.array(tr_list, dtype=float)
    sig = np.array(signal_list, dtype=float)

    # filtrer bort NaN/inf og ikke-positive signaler
    valid = np.isfinite(tr) & np.isfinite(sig) & (sig > 0)
    tr = tr[valid]
    sig = sig[valid]

    if len(tr) < 3:
        raise RuntimeError("For få datapunkt til å fitte T1!")

    # init-gjetning
    s0_init = sig.max()
    if tr.max() > tr.min():
        t1_init = (tr.max() - tr.min()) / 2.0
    else:
        t1_init = 1000.0  # ms, helt ok default

    popt, pcov = curve_fit(t1_model, tr, sig, p0=[s0_init, t1_init], maxfev=10000)
    s0, t1 = popt
    s0_err, t1_err = np.sqrt(np.diag(pcov))

    # R^2
    fit_vals = t1_model(tr, s0, t1)
    residuals = sig - fit_vals
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return s0, t1, s0_err, t1_err, r2


def main():
    # Samlere for Excel
    t1_results_rows = []       # til T1_Results (én rad per scan)
    metadata_rows = []         # til Metadata (én rad per scan)
    scan_signal_map = {}       # til DataPoints: {scan_no: {TR: mean_signal}}
    all_tr_values = set()      # alle TR_ms som dukker opp i noen scan

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(
            BASE_FOLDER, str(scan_no), "pdata", "1", "dicom"
        )
        print(f"\n[SCAN {scan_no}] Mappe: {dicom_folder}")

        images, repetition_times, ds0 = load_dicom_series(dicom_folder)
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

        # Mean signal per TR (RÅDATA – brukes til fitting og Excel)
        mean_signals = []
        tr_to_sig = {}
        for tr, img in zip(repetition_times, images):
            val = float(np.mean(img[mask]))
            mean_signals.append(val)
            tr_to_sig[float(tr)] = val
            all_tr_values.add(float(tr))
            print(f"[SCAN {scan_no}] TR={tr} ms, mean_signal={val:.2f}")

        scan_signal_map[scan_no] = tr_to_sig

        # Normalisering for plotting
        mean_signals_arr = np.array(mean_signals, dtype=float)
        if np.nanmax(mean_signals_arr) > 0:
            norm_signals_arr = mean_signals_arr / np.nanmax(mean_signals_arr)
        else:
            norm_signals_arr = np.full_like(mean_signals_arr, np.nan)

        # Fit T1 på RÅDATA (ikke normalisert)
        try:
            s0, t1, s0_err, t1_err, r2 = fit_t1_and_r2(repetition_times, mean_signals)
            print(f"[SCAN {scan_no}] T1 = {t1:.2f} ± {t1_err:.2f} ms, R² = {r2:.4f}")
        except Exception as e:
            print(f"[SCAN {scan_no}] ADVARSEL: klarte ikke å fitte T1: {e}")
            s0 = t1 = s0_err = t1_err = r2 = np.nan

        # ---------- PNG med ROI ----------
        fig, ax = plt.subplots()
        ax.imshow(img0, cmap="gray")

        dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
        for (cx_mm, cy_mm) in CIRCLE_CENTERS_MM:
            cx_p = cx_mm / dx_mm
            cy_p = cy_mm / dy_mm
            radius_p = CIRCLE_RADIUS_MM / dx_mm
            circ = plt.Circle((cx_p, cy_p), radius_p, fill=False, linewidth=2)
            ax.add_patch(circ)

        ax.set_title(f"Scan {scan_no} – T1 ROI")
        ax.axis("off")
        png_name = f"T1_mmCircleROI_scan{scan_no}.png"
        out_png = os.path.join(dicom_folder, png_name)
        plt.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[SCAN {scan_no}] Lagret ROI-PNG: {out_png}")

        # ---------- PNG med T1-plot (normalisert) ----------
        try:
            tr_arr = np.array(repetition_times, dtype=float)
            fig, ax = plt.subplots()
            ax.scatter(tr_arr, norm_signals_arr, label="Data (norm)", s=30)

            if np.isfinite(t1) and np.isfinite(s0) and s0 > 0:
                tr_fit = np.linspace(np.nanmin(tr_arr), np.nanmax(tr_arr), 200)
                fit_vals = t1_model(tr_fit, s0, t1)
                norm_fit_vals = fit_vals / s0
                ax.plot(tr_fit, norm_fit_vals, label="Fit (norm)", linewidth=2)

            ax.set_xlabel("TR (ms)")
            ax.set_ylabel("Normalisert signal")
            ax.set_title(f"T1-recovery – Scan {scan_no}")
            ax.grid(True)
            ax.legend()

            plot_name = f"T1_fit_normalized_scan{scan_no}.png"
            out_plot = os.path.join(dicom_folder, plot_name)
            plt.tight_layout()
            fig.savefig(out_plot, dpi=200)
            plt.close(fig)
            print(f"[SCAN {scan_no}] Lagret T1-plot-PNG: {out_plot}")
        except Exception as e:
            print(f"[SCAN {scan_no}] Klarte ikke å lage T1-plot: {e}")

        # ---------- Bygg rader til Excel ----------

        # 1) T1_Results
        t1_results_rows.append(
            {
                "Scan_no": scan_no,
                "T1_ms": t1,
                "T1_error_ms": t1_err,
                "R2": r2,
            }
        )

        # 2) Metadata
        valid_tr = [float(tr) for tr in repetition_times if np.isfinite(tr)]
        tr_list_str = ",".join(str(tr) for tr in valid_tr)

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
                "TR_list_ms": tr_list_str,
            }
        )

    # =====================================================
    # Etter løkken: bygg DataFrames og skriv til Excel
    # =====================================================

    # T1_Results-sheet
    df_t1 = pd.DataFrame(t1_results_rows, columns=["Scan_no", "T1_ms", "T1_error_ms", "R2"])

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
        "TR_list_ms",
    ]
    df_meta = pd.DataFrame(metadata_rows, columns=meta_columns)

    # DataPoints-sheet – TR_ms nedover, RÅSIGNAL per scan
    all_tr_sorted = sorted(all_tr_values)
    dp_raw = {"TR_ms": all_tr_sorted}
    dp_norm = {"TR_ms": all_tr_sorted}

    for scan_no, tr_to_sig in scan_signal_map.items():
        # Råsignal
        raw_vals = [tr_to_sig.get(tr, np.nan) for tr in all_tr_sorted]
        col_raw = f"Scan_{scan_no}"
        dp_raw[col_raw] = raw_vals

        # Normalisert signal
        raw_arr = np.array(raw_vals, dtype=float)
        if np.nanmax(raw_arr) > 0:
            norm_arr = raw_arr / np.nanmax(raw_arr)
        else:
            norm_arr = np.full_like(raw_arr, np.nan)
        col_norm = f"Scan_{scan_no}_norm"
        dp_norm[col_norm] = norm_arr

    df_dp_raw = pd.DataFrame(dp_raw)
    df_dp_norm = pd.DataFrame(dp_norm)

    # Skriv til én Excel-fil
    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_t1.to_excel(writer, index=False, sheet_name="T1_Results")
        df_dp_raw.to_excel(writer, index=False, sheet_name="DataPoints")
        df_dp_norm.to_excel(writer, index=False, sheet_name="DataPoints_norm")
        df_meta.to_excel(writer, index=False, sheet_name="Metadata")

    print(f"\n[OUTPUT] Lagret samlet T1-Excel-fil: {out_xlsx}")


if __name__ == "__main__":
    main()
