#!/usr/bin/env python3
"""
H2_T2_ax_with_SNR.py

T2-analyse for H2 axiale målinger, med ekstra beregning av SNR.

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
# KONFIGURASJON
# =====================================================

BASE_FOLDER = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\H2_Characterization"
SCAN_NUMBERS = [22, 29, 34, 39, 43, 47, 51, 55]

# Signal-ROI i mm
CIRCLE_CENTERS_MM = [(24.5, 22)]
CIRCLE_RADIUS_MM = 4.5

# Noise-ROI i mm. Flytt denne hvis den havner over prøve/celle/artefakter.
# Verdiene under legger en liten boks nær øvre venstre hjørne.
NOISE_RECT_LEFT_MM = 2.0
NOISE_RECT_TOP_MM = 2.0
NOISE_RECT_WIDTH_MM = 8.0
NOISE_RECT_HEIGHT_MM = 8.0

OUTPUT_EXCEL = "T2_H2_ax_scans_results_with_SNR.xlsx"

# =====================================================


def t2_model(te, s0, t2, c):
    return s0 * np.exp(-te / t2) + c


def load_dicom_series(dicom_folder):
    files = [
        os.path.join(dicom_folder, f)
        for f in os.listdir(dicom_folder)
        if f.lower().endswith(".dcm")
    ]
    if not files:
        raise FileNotFoundError(f"Ingen DICOM-filer i {dicom_folder}")

    datasets = [pydicom.dcmread(f) for f in files]
    datasets.sort(key=lambda ds: getattr(ds, "EchoTime", getattr(ds, "InstanceNumber", 0)))

    images = []
    echo_times = []

    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))

        images.append(arr)
        echo_times.append(float(getattr(ds, "EchoTime", np.nan)))

    return images, echo_times, datasets[0]


def create_circle_mask_mm(shape, pixel_spacing, centers_mm, radius_mm):
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
    te = np.array(te_list, dtype=float)
    sig = np.array(signal_list, dtype=float)

    threshold = 0.1 * np.nanmax(sig)
    valid = np.isfinite(te) & np.isfinite(sig) & (sig > threshold)

    te = te[valid]
    sig = sig[valid]

    if len(te) < 3:
        raise RuntimeError("For få datapunkt til å fitte T2!")

    s0_init = sig.max()
    t2_init = 10.0

    popt, pcov = curve_fit(
        t2_model,
        te,
        sig,
        p0=[s0_init, t2_init, 0],
        maxfev=10000,
    )

    s0, t2, c = popt
    s0_err, t2_err, c_err = np.sqrt(np.diag(pcov))

    fit_vals = t2_model(te, s0, t2, c)
    residuals = sig - fit_vals
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return s0, t2, c, s0_err, t2_err, c_err, r2


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
    t2_results_rows = []
    scan_signal_map = {}
    snr_simple_map = {}
    snr_corrected_map = {}
    noise_std_map = {}
    all_te_values = set()

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(BASE_FOLDER, str(scan_no), "pdata", "1", "dicom")
        print(f"\n[SCAN {scan_no}] {dicom_folder}")

        images, echo_times, ds0 = load_dicom_series(dicom_folder)
        img0 = images[0]
        rows, cols = img0.shape

        pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])

        signal_mask = create_circle_mask_mm(
            img0.shape,
            pixel_spacing,
            CIRCLE_CENTERS_MM,
            CIRCLE_RADIUS_MM,
        )
        noise_mask = create_rectangle_mask_mm(
            img0.shape,
            pixel_spacing,
            NOISE_RECT_LEFT_MM,
            NOISE_RECT_TOP_MM,
            NOISE_RECT_WIDTH_MM,
            NOISE_RECT_HEIGHT_MM,
        )

        overlap_px = int(np.sum(signal_mask & noise_mask))
        if overlap_px > 0:
            print(f"[SCAN {scan_no}] ADVARSEL: Signal-ROI og noise-ROI overlapper med {overlap_px} pixler.")

        print(f"[SCAN {scan_no}] Signal ROI pixler: {int(signal_mask.sum())}, Noise ROI pixler: {int(noise_mask.sum())}")

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

        try:
            s0, t2, c, s0_err, t2_err, c_err, r2 = fit_t2_and_r2(echo_times, mean_signals)
            print(f"T2 = {t2:.2f} ms | R² = {r2:.3f}")
        except Exception as e:
            print("FIT FAIL:", e)
            s0 = t2 = c = s0_err = t2_err = c_err = r2 = np.nan

        first_snr = snr_rows_this_scan[0] if snr_rows_this_scan else {}
        snr_simple_values = np.array([r["SNR_simple"] for r in snr_rows_this_scan], dtype=float)
        snr_corr_values = np.array([r["SNR_corrected_0p655"] for r in snr_rows_this_scan], dtype=float)

        t2_results_rows.append({
            "Scan_no": scan_no,
            "T2_ms": t2,
            "T2_error_ms": t2_err,
            "Offset_c": c,
            "Offset_c_error": c_err,
            "R2": r2,
            "Signal_mean_first_TE": first_snr.get("Signal_mean", np.nan),
            "Noise_mean_first_TE": first_snr.get("Noise_mean", np.nan),
            "Noise_std_first_TE": first_snr.get("Noise_std", np.nan),
            "SNR_simple_first_TE": first_snr.get("SNR_simple", np.nan),
            "SNR_corrected_first_TE": first_snr.get("SNR_corrected_0p655", np.nan),
            "SNR_simple_mean_all_TE": np.nanmean(snr_simple_values) if snr_simple_values.size else np.nan,
            "SNR_corrected_mean_all_TE": np.nanmean(snr_corr_values) if snr_corr_values.size else np.nan,
        })

        # PNG som viser både signal-ROI og noise-ROI
        fig, ax = plt.subplots()
        ax.imshow(img0, cmap="gray")
        add_circle_patch(ax, pixel_spacing)
        add_noise_rect_patch(ax, pixel_spacing)
        ax.set_title(f"Scan {scan_no} – signal ROI og noise ROI")
        ax.axis("off")
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=True)

        png_name = f"T2_H2_ROI_SNR_scan{scan_no}.png"
        out_png = os.path.join(dicom_folder, png_name)
        plt.tight_layout()
        fig.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close(fig)
        print(f"[SCAN {scan_no}] Lagret ROI/SNR-PNG: {out_png}")

    # =====================================================
    # DATAPOINTS + SNR sheets
    # =====================================================

    all_te_sorted = sorted(all_te_values)

    dp_raw = {"TE_ms": all_te_sorted}
    dp_norm = {"TE_ms": all_te_sorted}
    dp_snr_simple = {"TE_ms": all_te_sorted}
    dp_snr_corrected = {"TE_ms": all_te_sorted}
    dp_noise_std = {"TE_ms": all_te_sorted}

    for scan_no, te_to_sig in scan_signal_map.items():
        raw_vals = [te_to_sig.get(te, np.nan) for te in all_te_sorted]
        dp_raw[f"Scan_{scan_no}"] = raw_vals

        raw_arr = np.array(raw_vals, dtype=float)
        if np.nanmax(raw_arr) > 0:
            norm_arr = raw_arr / np.nanmax(raw_arr)
        else:
            norm_arr = np.full_like(raw_arr, np.nan)

        dp_norm[f"Scan_{scan_no}_norm"] = norm_arr
        dp_snr_simple[f"Scan_{scan_no}_SNR"] = [snr_simple_map[scan_no].get(te, np.nan) for te in all_te_sorted]
        dp_snr_corrected[f"Scan_{scan_no}_SNR_corr"] = [snr_corrected_map[scan_no].get(te, np.nan) for te in all_te_sorted]
        dp_noise_std[f"Scan_{scan_no}_noise_std"] = [noise_std_map[scan_no].get(te, np.nan) for te in all_te_sorted]

    df_results = pd.DataFrame(t2_results_rows)
    df_dp_raw = pd.DataFrame(dp_raw)
    df_dp_norm = pd.DataFrame(dp_norm)
    df_snr_simple = pd.DataFrame(dp_snr_simple)
    df_snr_corrected = pd.DataFrame(dp_snr_corrected)
    df_noise_std = pd.DataFrame(dp_noise_std)

    # =====================================================
    # EXCEL OUTPUT
    # =====================================================

    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, index=False, sheet_name="T2_Results")
        df_dp_raw.to_excel(writer, index=False, sheet_name="DataPoints")
        df_dp_norm.to_excel(writer, index=False, sheet_name="DataPoints_norm")
        df_snr_simple.to_excel(writer, index=False, sheet_name="SNR_simple")
        df_snr_corrected.to_excel(writer, index=False, sheet_name="SNR_corrected")
        df_noise_std.to_excel(writer, index=False, sheet_name="Noise_std")

    print(f"\n[OUTPUT] Lagret: {out_xlsx}")


if __name__ == "__main__":
    main()
