#!/usr/bin/env python3
"""
CH4_T2_ax_with_SNR.py

T2-analyse for CH4 axiale målinger, med ekstra beregning av SNR.

SNR beregnes fra:
    SNR_simple    = mean(signal ROI) / std(noise ROI)
    SNR_corrected = 0.655 * mean(signal ROI) / std(noise ROI)

Signal-ROI er den eksisterende sirkulære ROI-en i prøven.
Noise-ROI er et lite rektangel i bakgrunnen. Juster NOISE_RECT_* dersom
rektangelet havner over prøve, celle, ghosting eller tydelige artefakter.
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

# Hvilke scan-mapper vil du prosessere (under BASE_FOLDER)?
SCAN_NUMBERS = [7, 17, 26, 35, 44, 53, 62, 71, 80, 89, 98]

# Signal-ROI i mm (samme for alle scan)
CIRCLE_CENTERS_MM = [(29.5, 29.75)]
CIRCLE_RADIUS_MM = 13.0

# Noise-ROI i mm. Flytt denne hvis den havner over prøve/celle/artefakter.
# Verdiene under legger en liten boks nær øvre venstre hjørne.
NOISE_RECT_LEFT_MM = 2.0
NOISE_RECT_TOP_MM = 2.0
NOISE_RECT_WIDTH_MM = 8.0
NOISE_RECT_HEIGHT_MM = 8.0

# Output Excel-fil (samlet for alle scan)
OUTPUT_EXCEL = "T2_all_scans_results_with_SNR.xlsx"

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
        radius_p = radius_mm / dx_mm
        dist2 = (X - cx_p) ** 2 + (Y - cy_p) ** 2
        mask_total |= (dist2 <= radius_p ** 2)

    return mask_total


def create_rectangle_mask_mm(shape, pixel_spacing, left_mm, top_mm, width_mm, height_mm):
    """Lager en rektangulær maske gitt i mm fra venstre/toppen av bildet."""
    rows, cols = shape
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])

    left_px = int(round(left_mm / dx_mm))
    right_px = int(round((left_mm + width_mm) / dx_mm))
    top_px = int(round(top_mm / dy_mm))
    bottom_px = int(round((top_mm + height_mm) / dy_mm))

    # Klipp til bildekantene slik at koden ikke krasjer hvis ROI er litt utenfor.
    left_px = max(0, min(cols, left_px))
    right_px = max(0, min(cols, right_px))
    top_px = max(0, min(rows, top_px))
    bottom_px = max(0, min(rows, bottom_px))

    mask = np.zeros(shape, dtype=bool)
    mask[top_px:bottom_px, left_px:right_px] = True
    return mask


def compute_snr(img, signal_mask, noise_mask):
    signal_mean = float(np.mean(img[signal_mask])) if np.any(signal_mask) else np.nan
    noise_mean = float(np.mean(img[noise_mask])) if np.any(noise_mask) else np.nan
    noise_std = float(np.std(img[noise_mask], ddof=1)) if np.sum(noise_mask) > 1 else np.nan

    if np.isfinite(noise_std) and noise_std > 0:
        snr_simple = signal_mean / noise_std
        snr_corrected = 0.655 * signal_mean / noise_std
    else:
        snr_simple = np.nan
        snr_corrected = np.nan

    return signal_mean, noise_mean, noise_std, snr_simple, snr_corrected


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


def add_circle_patch(ax, pixel_spacing):
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    for (cx_mm, cy_mm) in CIRCLE_CENTERS_MM:
        cx_p = cx_mm / dx_mm
        cy_p = cy_mm / dy_mm
        radius_p = CIRCLE_RADIUS_MM / dx_mm
        circ = plt.Circle(
            (cx_p, cy_p),
            radius_p,
            fill=False,
            linewidth=2,
            edgecolor="yellow",
            label="Signal ROI",
        )
        ax.add_patch(circ)


def add_noise_rect_patch(ax, pixel_spacing):
    dy_mm, dx_mm = float(pixel_spacing[0]), float(pixel_spacing[1])
    left_px = NOISE_RECT_LEFT_MM / dx_mm
    top_px = NOISE_RECT_TOP_MM / dy_mm
    width_px = NOISE_RECT_WIDTH_MM / dx_mm
    height_px = NOISE_RECT_HEIGHT_MM / dy_mm

    rect = Rectangle(
        (left_px, top_px),
        width_px,
        height_px,
        fill=False,
        linewidth=2,
        edgecolor="cyan",
        label="Noise ROI",
    )
    ax.add_patch(rect)


def main():
    # Samlere for Excel
    t2_results_rows = []
    metadata_rows = []
    scan_signal_map = {}
    snr_simple_map = {}
    snr_corrected_map = {}
    noise_std_map = {}
    all_te_values = set()

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

        # Signal-ROI og noise-ROI
        signal_mask = create_circle_mask_mm(
            shape=img0.shape,
            pixel_spacing=pixel_spacing,
            centers_mm=CIRCLE_CENTERS_MM,
            radius_mm=CIRCLE_RADIUS_MM,
        )
        noise_mask = create_rectangle_mask_mm(
            shape=img0.shape,
            pixel_spacing=pixel_spacing,
            left_mm=NOISE_RECT_LEFT_MM,
            top_mm=NOISE_RECT_TOP_MM,
            width_mm=NOISE_RECT_WIDTH_MM,
            height_mm=NOISE_RECT_HEIGHT_MM,
        )

        overlap_px = int(np.sum(signal_mask & noise_mask))
        if overlap_px > 0:
            print(f"[SCAN {scan_no}] ADVARSEL: Signal-ROI og noise-ROI overlapper med {overlap_px} pixler.")

        roi_area_px = int(signal_mask.sum())
        roi_fraction = roi_area_px / (rows * cols) * 100.0
        noise_area_px = int(noise_mask.sum())
        noise_fraction = noise_area_px / (rows * cols) * 100.0
        print(f"[SCAN {scan_no}] Signal ROI pixler: {roi_area_px}, {roi_fraction:.2f}% av bildet")
        print(f"[SCAN {scan_no}] Noise ROI pixler: {noise_area_px}, {noise_fraction:.2f}% av bildet")

        # Mean signal og SNR per TE
        mean_signals = []
        te_to_sig = {}
        te_to_snr_simple = {}
        te_to_snr_corrected = {}
        te_to_noise_std = {}
        snr_rows_this_scan = []

        for te, img in zip(echo_times, images):
            signal_mean, noise_mean, noise_std, snr_simple, snr_corrected = compute_snr(img, signal_mask, noise_mask)

            mean_signals.append(signal_mean)
            te_to_sig[float(te)] = signal_mean
            te_to_snr_simple[float(te)] = snr_simple
            te_to_snr_corrected[float(te)] = snr_corrected
            te_to_noise_std[float(te)] = noise_std
            all_te_values.add(float(te))

            snr_rows_this_scan.append(
                {
                    "TE_ms": float(te),
                    "Signal_mean": signal_mean,
                    "Noise_mean": noise_mean,
                    "Noise_std": noise_std,
                    "SNR_simple": snr_simple,
                    "SNR_corrected_0p655": snr_corrected,
                }
            )

            print(
                f"[SCAN {scan_no}] TE={te} ms, signal={signal_mean:.2f}, "
                f"noise_std={noise_std:.2f}, SNR={snr_simple:.2f}, SNR_corr={snr_corrected:.2f}"
            )

        scan_signal_map[scan_no] = te_to_sig
        snr_simple_map[scan_no] = te_to_snr_simple
        snr_corrected_map[scan_no] = te_to_snr_corrected
        noise_std_map[scan_no] = te_to_noise_std

        # Fit T2
        try:
            s0, t2, s0_err, t2_err, r2 = fit_t2_and_r2(echo_times, mean_signals)
            print(f"[SCAN {scan_no}] T2 = {t2:.2f} ± {t2_err:.2f} ms, R² = {r2:.4f}")
        except Exception as e:
            print(f"[SCAN {scan_no}] ADVARSEL: klarte ikke å fitte T2: {e}")
            s0 = t2 = s0_err = t2_err = r2 = np.nan

        first_snr = snr_rows_this_scan[0] if snr_rows_this_scan else {}
        snr_simple_values = np.array([r["SNR_simple"] for r in snr_rows_this_scan], dtype=float)
        snr_corr_values = np.array([r["SNR_corrected_0p655"] for r in snr_rows_this_scan], dtype=float)

        # PNG med både signal-ROI og noise-ROI
        fig, ax = plt.subplots()
        ax.imshow(img0, cmap="gray")
        add_circle_patch(ax, pixel_spacing)
        add_noise_rect_patch(ax, pixel_spacing)
        ax.set_title(f"Scan {scan_no} – signal ROI og noise ROI")
        ax.axis("off")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True)
        png_name = f"T2_CH4_ROI_SNR_scan{scan_no}.png"
        out_png = os.path.join(dicom_folder, png_name)
        plt.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SCAN {scan_no}] Lagret ROI/SNR-PNG: {out_png}")

        # ---------- Bygg rader til Excel ----------

        t2_results_rows.append(
            {
                "Scan_no": scan_no,
                "T2_ms": t2,
                "T2_error_ms": t2_err,
                "R2": r2,
                "Signal_mean_first_TE": first_snr.get("Signal_mean", np.nan),
                "Noise_mean_first_TE": first_snr.get("Noise_mean", np.nan),
                "Noise_std_first_TE": first_snr.get("Noise_std", np.nan),
                "SNR_simple_first_TE": first_snr.get("SNR_simple", np.nan),
                "SNR_corrected_first_TE": first_snr.get("SNR_corrected_0p655", np.nan),
                "SNR_simple_mean_all_TE": np.nanmean(snr_simple_values) if snr_simple_values.size else np.nan,
                "SNR_corrected_mean_all_TE": np.nanmean(snr_corr_values) if snr_corr_values.size else np.nan,
            }
        )

        # Metadata
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
                "Signal_ROI_centers_mm": str(CIRCLE_CENTERS_MM),
                "Signal_ROI_radius_mm": CIRCLE_RADIUS_MM,
                "Noise_ROI_left_mm": NOISE_RECT_LEFT_MM,
                "Noise_ROI_top_mm": NOISE_RECT_TOP_MM,
                "Noise_ROI_width_mm": NOISE_RECT_WIDTH_MM,
                "Noise_ROI_height_mm": NOISE_RECT_HEIGHT_MM,
            }
        )

    # =====================================================
    # Etter løkken: bygg DataFrames og skriv til Excel
    # =====================================================

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
        "Signal_ROI_centers_mm",
        "Signal_ROI_radius_mm",
        "Noise_ROI_left_mm",
        "Noise_ROI_top_mm",
        "Noise_ROI_width_mm",
        "Noise_ROI_height_mm",
    ]
    df_meta = pd.DataFrame(metadata_rows, columns=meta_columns)

    all_te_sorted = sorted(all_te_values)
    dp_raw = {"TE_ms": all_te_sorted}
    dp_snr_simple = {"TE_ms": all_te_sorted}
    dp_snr_corrected = {"TE_ms": all_te_sorted}
    dp_noise_std = {"TE_ms": all_te_sorted}

    for scan_no, te_to_sig in scan_signal_map.items():
        dp_raw[f"Scan_{scan_no}"] = [te_to_sig.get(te, np.nan) for te in all_te_sorted]
        dp_snr_simple[f"Scan_{scan_no}_SNR"] = [snr_simple_map[scan_no].get(te, np.nan) for te in all_te_sorted]
        dp_snr_corrected[f"Scan_{scan_no}_SNR_corr"] = [snr_corrected_map[scan_no].get(te, np.nan) for te in all_te_sorted]
        dp_noise_std[f"Scan_{scan_no}_noise_std"] = [noise_std_map[scan_no].get(te, np.nan) for te in all_te_sorted]

    df_dp = pd.DataFrame(dp_raw)
    df_snr_simple = pd.DataFrame(dp_snr_simple)
    df_snr_corrected = pd.DataFrame(dp_snr_corrected)
    df_noise_std = pd.DataFrame(dp_noise_std)

    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_t2.to_excel(writer, index=False, sheet_name="T2_Results")
        df_dp.to_excel(writer, index=False, sheet_name="DataPoints")
        df_snr_simple.to_excel(writer, index=False, sheet_name="SNR_simple")
        df_snr_corrected.to_excel(writer, index=False, sheet_name="SNR_corrected")
        df_noise_std.to_excel(writer, index=False, sheet_name="Noise_std")
        df_meta.to_excel(writer, index=False, sheet_name="Metadata")

    print(f"\n[OUTPUT] Lagret samlet Excel-fil: {out_xlsx}")


if __name__ == "__main__":
    main()
