#!/usr/bin/env python3
"""
CH4_T1_ax_with_SNR.py

T1-analyse for CH4 axiale målinger, med ekstra beregning av SNR.

SNR beregnes fra:
    SNR_simple    = mean(signal ROI) / std(noise ROI)
    SNR_corrected = 0.655 * mean(signal ROI) / std(noise ROI)

Signal-ROI er den eksisterende sirkulære ROI-en i prøven.
Noise-ROI er et lite rektangel i bakgrunnen. Juster NOISE_RECT_* dersom
rektangelet havner over prøve, celle, ghosting eller tydelige artefakter.

For T1 er signalet lavest ved korte TR og høyest ved lange TR. Derfor lagres
både SNR for alle TR-punkter i egne Excel-ark, og en oppsummering ved TR-punktet
med høyest signal i T1_Results.
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
SCAN_NUMBERS = [9, 19, 28, 37, 46, 55, 64, 73, 82, 91, 100, 105, 106, 107]

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
OUTPUT_EXCEL = "T1_all_scans_results_with_SNR.xlsx"

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
    t1_results_rows = []
    metadata_rows = []
    scan_signal_map = {}
    snr_simple_map = {}
    snr_corrected_map = {}
    noise_std_map = {}
    noise_mean_map = {}
    all_tr_values = set()

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

        roi_area_px = int(signal_mask.sum())
        noise_area_px = int(noise_mask.sum())
        roi_fraction = roi_area_px / (rows * cols) * 100.0
        noise_fraction = noise_area_px / (rows * cols) * 100.0
        print(f"[SCAN {scan_no}] Signal-ROI pixler: {roi_area_px}, {roi_fraction:.2f}% av bildet")
        print(f"[SCAN {scan_no}] Noise-ROI pixler: {noise_area_px}, {noise_fraction:.2f}% av bildet")

        # Mean signal og SNR per TR (RÅDATA – brukes til fitting og Excel)
        mean_signals = []
        tr_to_sig = {}
        tr_to_snr_simple = {}
        tr_to_snr_corrected = {}
        tr_to_noise_std = {}
        tr_to_noise_mean = {}
        snr_rows_this_scan = []

        for tr, img in zip(repetition_times, images):
            signal_mean, noise_mean, noise_std, snr_simple, snr_corrected = compute_snr(
                img, signal_mask, noise_mask
            )

            mean_signals.append(signal_mean)
            tr_float = float(tr)
            tr_to_sig[tr_float] = signal_mean
            tr_to_snr_simple[tr_float] = snr_simple
            tr_to_snr_corrected[tr_float] = snr_corrected
            tr_to_noise_std[tr_float] = noise_std
            tr_to_noise_mean[tr_float] = noise_mean
            all_tr_values.add(tr_float)

            snr_rows_this_scan.append(
                {
                    "TR_ms": tr_float,
                    "Signal_mean": signal_mean,
                    "Noise_mean": noise_mean,
                    "Noise_std": noise_std,
                    "SNR_simple": snr_simple,
                    "SNR_corrected": snr_corrected,
                }
            )

            print(
                f"[SCAN {scan_no}] TR={tr_float} ms, "
                f"signal={signal_mean:.2f}, noise_std={noise_std:.2f}, "
                f"SNRcorr={snr_corrected:.2f}"
            )

        scan_signal_map[scan_no] = tr_to_sig
        snr_simple_map[scan_no] = tr_to_snr_simple
        snr_corrected_map[scan_no] = tr_to_snr_corrected
        noise_std_map[scan_no] = tr_to_noise_std
        noise_mean_map[scan_no] = tr_to_noise_mean

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

        # Velg TR-punktet med høyest signal som representativ SNR-oppsummering for T1.
        signal_values = np.array([r["Signal_mean"] for r in snr_rows_this_scan], dtype=float)
        if signal_values.size and np.any(np.isfinite(signal_values)):
            idx_max_signal = int(np.nanargmax(signal_values))
            max_signal_snr = snr_rows_this_scan[idx_max_signal]
        else:
            max_signal_snr = {}

        snr_simple_values = np.array([r["SNR_simple"] for r in snr_rows_this_scan], dtype=float)
        snr_corrected_values = np.array([r["SNR_corrected"] for r in snr_rows_this_scan], dtype=float)

        # ---------- PNG med signal-ROI og noise-ROI ----------
        fig, ax = plt.subplots()
        ax.imshow(img0, cmap="gray")
        add_circle_patch(ax, pixel_spacing)
        add_noise_rect_patch(ax, pixel_spacing)
        ax.set_title(f"Scan {scan_no} – T1 signal ROI og noise ROI")
        ax.axis("off")
        ax.legend(loc="lower right")
        png_name = f"T1_ax_SNR_ROI_scan{scan_no}.png"
        out_png = os.path.join(dicom_folder, png_name)
        plt.tight_layout()
        fig.savefig(out_png, dpi=200)
        plt.close(fig)
        print(f"[SCAN {scan_no}] Lagret ROI/SNR-PNG: {out_png}")

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
                "S0": s0,
                "S0_error": s0_err,
                "TR_max_signal_ms": max_signal_snr.get("TR_ms", np.nan),
                "Signal_mean_max_signal_TR": max_signal_snr.get("Signal_mean", np.nan),
                "Noise_std_max_signal_TR": max_signal_snr.get("Noise_std", np.nan),
                "SNR_simple_max_signal_TR": max_signal_snr.get("SNR_simple", np.nan),
                "SNR_corrected_max_signal_TR": max_signal_snr.get("SNR_corrected", np.nan),
                "SNR_simple_mean_all_TR": np.nanmean(snr_simple_values) if snr_simple_values.size else np.nan,
                "SNR_corrected_mean_all_TR": np.nanmean(snr_corrected_values) if snr_corrected_values.size else np.nan,
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

    # T1_Results-sheet
    df_t1 = pd.DataFrame(t1_results_rows)

    # Metadata-sheet
    df_meta = pd.DataFrame(metadata_rows)

    # DataPoints og SNR-sheets – TR_ms nedover, én kolonne per scan
    all_tr_sorted = sorted(all_tr_values)
    dp_raw = {"TR_ms": all_tr_sorted}
    dp_norm = {"TR_ms": all_tr_sorted}
    dp_snr_simple = {"TR_ms": all_tr_sorted}
    dp_snr_corrected = {"TR_ms": all_tr_sorted}
    dp_noise_std = {"TR_ms": all_tr_sorted}
    dp_noise_mean = {"TR_ms": all_tr_sorted}

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

        # SNR og noise
        dp_snr_simple[f"Scan_{scan_no}"] = [snr_simple_map[scan_no].get(tr, np.nan) for tr in all_tr_sorted]
        dp_snr_corrected[f"Scan_{scan_no}"] = [snr_corrected_map[scan_no].get(tr, np.nan) for tr in all_tr_sorted]
        dp_noise_std[f"Scan_{scan_no}"] = [noise_std_map[scan_no].get(tr, np.nan) for tr in all_tr_sorted]
        dp_noise_mean[f"Scan_{scan_no}"] = [noise_mean_map[scan_no].get(tr, np.nan) for tr in all_tr_sorted]

    df_dp_raw = pd.DataFrame(dp_raw)
    df_dp_norm = pd.DataFrame(dp_norm)
    df_snr_simple = pd.DataFrame(dp_snr_simple)
    df_snr_corrected = pd.DataFrame(dp_snr_corrected)
    df_noise_std = pd.DataFrame(dp_noise_std)
    df_noise_mean = pd.DataFrame(dp_noise_mean)

    # Skriv til én Excel-fil
    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)
    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_t1.to_excel(writer, index=False, sheet_name="T1_Results")
        df_dp_raw.to_excel(writer, index=False, sheet_name="DataPoints")
        df_dp_norm.to_excel(writer, index=False, sheet_name="DataPoints_norm")
        df_snr_simple.to_excel(writer, index=False, sheet_name="SNR_simple")
        df_snr_corrected.to_excel(writer, index=False, sheet_name="SNR_corrected")
        df_noise_std.to_excel(writer, index=False, sheet_name="Noise_std")
        df_noise_mean.to_excel(writer, index=False, sheet_name="Noise_mean")
        df_meta.to_excel(writer, index=False, sheet_name="Metadata")

    print(f"\n[OUTPUT] Lagret samlet T1-Excel-fil: {out_xlsx}")


if __name__ == "__main__":
    main()
