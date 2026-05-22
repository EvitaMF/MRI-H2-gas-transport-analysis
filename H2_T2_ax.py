#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# =====================================================
# KONFIGURASJON
# =====================================================

BASE_FOLDER = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\H2_Characterization"
SCAN_NUMBERS = [22, 29, 34, 39, 43, 47, 51, 55]

CIRCLE_CENTERS_MM = [(24.5, 22)]
CIRCLE_RADIUS_MM = 4.5

OUTPUT_EXCEL = "T2_H2_ax_scans_results.xlsx"

# =====================================================

def t2_model(te, s0, t2, c):
    return s0 * np.exp(-te / t2) + c


def load_dicom_series(dicom_folder):
    files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.lower().endswith(".dcm")]
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
        maxfev=10000
    )

    s0, t2, c = popt
    s0_err, t2_err, c_err = np.sqrt(np.diag(pcov))

    fit_vals = t2_model(te, s0, t2, c)
    residuals = sig - fit_vals
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return s0, t2, c, s0_err, t2_err, r2


def main():
    t2_results_rows = []
    scan_signal_map = {}
    all_te_values = set()

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(BASE_FOLDER, str(scan_no), "pdata", "1", "dicom")
        print(f"\n[SCAN {scan_no}] {dicom_folder}")

        images, echo_times, ds0 = load_dicom_series(dicom_folder)
        img0 = images[0]

        pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])
        mask = create_circle_mask_mm(img0.shape, pixel_spacing, CIRCLE_CENTERS_MM, CIRCLE_RADIUS_MM)

        mean_signals = []
        te_to_sig = {}

        for te, img in zip(echo_times, images):
            val = float(np.mean(img[mask]))
            mean_signals.append(val)
            te_to_sig[float(te)] = val
            all_te_values.add(float(te))

        scan_signal_map[scan_no] = te_to_sig

        try:
            s0, t2, c, s0_err, t2_err, r2 = fit_t2_and_r2(echo_times, mean_signals)
            print(f"T2 = {t2:.2f} ms | R² = {r2:.3f}")
        except Exception as e:
            print("FIT FAIL:", e)
            s0 = t2 = c = s0_err = t2_err = r2 = np.nan

        t2_results_rows.append({
            "Scan_no": scan_no,
            "T2_ms": t2,
            "T2_error_ms": t2_err,
            "R2": r2,
        })

    # =====================================================
    # 🔥 DATAPOINTS (samme som T1)
    # =====================================================

    all_te_sorted = sorted(all_te_values)

    dp_raw = {"TE_ms": all_te_sorted}
    dp_norm = {"TE_ms": all_te_sorted}

    for scan_no, te_to_sig in scan_signal_map.items():
        raw_vals = [te_to_sig.get(te, np.nan) for te in all_te_sorted]
        dp_raw[f"Scan_{scan_no}"] = raw_vals

        raw_arr = np.array(raw_vals, dtype=float)
        if np.nanmax(raw_arr) > 0:
            norm_arr = raw_arr / np.nanmax(raw_arr)
        else:
            norm_arr = np.full_like(raw_arr, np.nan)

        dp_norm[f"Scan_{scan_no}_norm"] = norm_arr

    df_results = pd.DataFrame(t2_results_rows)
    df_dp_raw = pd.DataFrame(dp_raw)
    df_dp_norm = pd.DataFrame(dp_norm)

    # =====================================================
    # 🔥 EXCEL OUTPUT
    # =====================================================

    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, index=False, sheet_name="T2_Results")
        df_dp_raw.to_excel(writer, index=False, sheet_name="DataPoints")
        df_dp_norm.to_excel(writer, index=False, sheet_name="DataPoints_norm")

    print(f"\n[OUTPUT] Lagret: {out_xlsx}")


if __name__ == "__main__":
    main()