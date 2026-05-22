#!/usr/bin/env python3
"""
T1_batch_mmRectROI_sagittal.py

- Kjører T1-analyse for sagittale målinger i samme mappe-struktur som axial-scriptet.
- Hvert scan ligger i: BASE_FOLDER/<scan_no>/pdata/1/dicom
- ROI: rektangel definert i millimeter fra bildekoordinater.

ROI-definisjon:
    RECT_TOP_MM      : avstand fra toppen av bildet til øvre kant av ROI [mm]
    RECT_HEIGHT_MM   : høyde på ROI [mm]
    RECT_WIDTH_MM    : bredde på ROI [mm], None = full bildebredde
    RECT_CENTER_X_MM : x-posisjon til senter av ROI [mm], None = midt i bildet

For hver scan:
    * Leser DICOM-serie og sorterer på RepetitionTime (TR)
    * Lager rektangulær ROI-mask fra mm -> pixler via PixelSpacing
    * Beregner gjennomsnittssignal i ROI for hver TR
    * Fitter T1 på råsignalet: S(TR) = S0 * (1 - exp(-TR/T1))
    * Beregner R²
    * Lager PNG med ROI på første bilde
    * Lager PNG med normaliserte datapunkter + normalisert fit

Etter alle scan:
    * Lager ett Excel-dokument med sheets:
        - T1_Results
        - DataPoints
        - DataPoints_norm
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

# Sagittale T1-scan: fyll inn riktige scan-numre her
SCAN_NUMBERS = []  # f.eks. [11, 21, 30, 39]

# Rektangulær ROI i mm
RECT_TOP_MM = 55        # mm fra toppen av bildet
RECT_HEIGHT_MM = 10     # mm høyde
RECT_WIDTH_MM = 7       # mm bredde, None = full bredde
RECT_CENTER_X_MM = None # None = midt i bildet

# Output Excel-fil
OUTPUT_EXCEL = "T1_sagittal_rectROI_results.xlsx"

# =====================================================


def t1_model(tr, s0, t1):
    """T1-funksjon: S(TR) = S0 * (1 - exp(-TR/T1))."""
    return s0 * (1.0 - np.exp(-tr / t1))


def load_dicom_series(dicom_folder):
    """
    Leser og sorterer DICOM-serie på RepetitionTime (TR), med InstanceNumber som fallback.
    Returnerer images, repetition_times og første dataset for metadata.
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

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr * slope + intercept

        images.append(arr)

        tr_val = getattr(ds, "RepetitionTime", np.nan)
        repetition_times.append(float(tr_val) if tr_val is not None else np.nan)

    return images, repetition_times, datasets[0]


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


def fit_t1_and_r2(tr_list, signal_list):
    """Fitter T1 og beregner R². Returnerer s0, t1, s0_err, t1_err, r2."""
    tr = np.array(tr_list, dtype=float)
    sig = np.array(signal_list, dtype=float)

    valid = np.isfinite(tr) & np.isfinite(sig) & (sig > 0)
    tr = tr[valid]
    sig = sig[valid]

    if len(tr) < 3:
        raise RuntimeError("For få datapunkt til å fitte T1!")

    s0_init = sig.max()
    t1_init = (tr.max() - tr.min()) / 2.0 if tr.max() > tr.min() else 1000.0

    popt, pcov = curve_fit(t1_model, tr, sig, p0=[s0_init, t1_init], maxfev=10000)
    s0, t1 = popt
    s0_err, t1_err = np.sqrt(np.diag(pcov))

    fit_vals = t1_model(tr, s0, t1)
    residuals = sig - fit_vals
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return s0, t1, s0_err, t1_err, r2


def main():
    if not SCAN_NUMBERS:
        raise ValueError("SCAN_NUMBERS er tom. Legg inn sagittale T1-scan før du kjører scriptet.")

    t1_results_rows = []
    metadata_rows = []
    scan_signal_map = {}
    all_tr_values = set()

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(BASE_FOLDER, str(scan_no), "pdata", "1", "dicom")
        print(f"\n[SCAN {scan_no}] Mappe: {dicom_folder}")

        images, repetition_times, ds0 = load_dicom_series(dicom_folder)
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
        tr_to_sig = {}
        for tr, img in zip(repetition_times, images):
            val = float(np.mean(img[mask]))
            mean_signals.append(val)
            tr_to_sig[float(tr)] = val
            all_tr_values.add(float(tr))
            print(f"[SCAN {scan_no}] TR={tr} ms, mean_signal={val:.2f}")

        scan_signal_map[scan_no] = tr_to_sig

        mean_signals_arr = np.array(mean_signals, dtype=float)
        if np.nanmax(mean_signals_arr) > 0:
            norm_signals_arr = mean_signals_arr / np.nanmax(mean_signals_arr)
        else:
            norm_signals_arr = np.full_like(mean_signals_arr, np.nan)

        try:
            s0, t1, s0_err, t1_err, r2 = fit_t1_and_r2(repetition_times, mean_signals)
            print(f"[SCAN {scan_no}] T1 = {t1:.2f} ± {t1_err:.2f} ms, R² = {r2:.4f}")
        except Exception as e:
            print(f"[SCAN {scan_no}] ADVARSEL: klarte ikke å fitte T1: {e}")
            s0 = t1 = s0_err = t1_err = r2 = np.nan

        # ---------- PNG med ROI ----------
        fig, ax = plt.subplots()
        ax.imshow(img0, cmap="gray")
        rect = Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.set_title(f"Scan {scan_no} – sagittal T1 ROI")
        ax.axis("off")

        png_name = f"T1_sagittal_mmRectROI_scan{scan_no}.png"
        out_png = os.path.join(dicom_folder, png_name)
        plt.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[SCAN {scan_no}] Lagret ROI-PNG: {out_png}")

        # ---------- PNG med T1-plot ----------
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
            ax.set_title(f"T1 recovery – sagittal scan {scan_no}")
            ax.grid(True)
            ax.legend()

            plot_name = f"T1_sagittal_fit_normalized_scan{scan_no}.png"
            out_plot = os.path.join(dicom_folder, plot_name)
            plt.tight_layout()
            fig.savefig(out_plot, dpi=200)
            plt.close(fig)
            print(f"[SCAN {scan_no}] Lagret T1-plot-PNG: {out_plot}")
        except Exception as e:
            print(f"[SCAN {scan_no}] Klarte ikke å lage T1-plot: {e}")

        t1_results_rows.append(
            {
                "Scan_no": scan_no,
                "T1_ms": t1,
                "T1_error_ms": t1_err,
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

    df_t1 = pd.DataFrame(t1_results_rows)

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

    all_tr_sorted = sorted(all_tr_values)
    dp_raw = {"TR_ms": all_tr_sorted}
    dp_norm = {"TR_ms": all_tr_sorted}

    for scan_no, tr_to_sig in scan_signal_map.items():
        raw_vals = [tr_to_sig.get(tr, np.nan) for tr in all_tr_sorted]
        dp_raw[f"Scan_{scan_no}"] = raw_vals

        raw_arr = np.array(raw_vals, dtype=float)
        if np.nanmax(raw_arr) > 0:
            norm_arr = raw_arr / np.nanmax(raw_arr)
        else:
            norm_arr = np.full_like(raw_arr, np.nan)
        dp_norm[f"Scan_{scan_no}_norm"] = norm_arr

    df_dp_raw = pd.DataFrame(dp_raw)
    df_dp_norm = pd.DataFrame(dp_norm)

    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_t1.to_excel(writer, index=False, sheet_name="T1_Results")
        df_dp_raw.to_excel(writer, index=False, sheet_name="DataPoints")
        df_dp_norm.to_excel(writer, index=False, sheet_name="DataPoints_norm")
        df_meta.to_excel(writer, index=False, sheet_name="Metadata")

    print(f"\n[OUTPUT] Lagret samlet T1-Excel-fil: {out_xlsx}")


if __name__ == "__main__":
    main()
