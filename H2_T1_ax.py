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
SCAN_NUMBERS = [23, 30, 35, 40, 44, 48, 52, 56, 59, 62, 66]

CIRCLE_CENTERS_MM = [(24.5, 22)]
CIRCLE_RADIUS_MM = 4.5

OUTPUT_EXCEL = "T1_H2_ax_scans_results.xlsx"

# =====================================================

def t1_model(tr, s0, t1, c):
    return s0 * (1.0 - np.exp(-tr / t1)) + c


def load_dicom_series(dicom_folder):
    files = [os.path.join(dicom_folder, f) for f in os.listdir(dicom_folder) if f.lower().endswith(".dcm")]
    datasets = [pydicom.dcmread(f) for f in files]

    datasets.sort(key=lambda ds: getattr(ds, "RepetitionTime", getattr(ds, "InstanceNumber", 0)))

    images = []
    repetition_times = []

    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        arr = arr * float(getattr(ds, "RescaleSlope", 1.0)) + float(getattr(ds, "RescaleIntercept", 0.0))

        images.append(arr)
        repetition_times.append(float(getattr(ds, "RepetitionTime", np.nan)))

    return images, repetition_times, datasets[0]


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


def fit_t1_and_r2(tr_list, signal_list):
    tr = np.array(tr_list, dtype=float)
    sig = np.array(signal_list, dtype=float)

    threshold = 0.05 * np.nanmax(sig)
    valid = np.isfinite(tr) & np.isfinite(sig) & (sig > threshold)

    tr = tr[valid]
    sig = sig[valid]

    if len(tr) < 3:
        raise RuntimeError("For få datapunkt til å fitte T1!")

    s0_init = sig.max()
    t1_init = (tr.max() - tr.min()) / 2 if tr.max() > tr.min() else 200.0

    popt, pcov = curve_fit(
        t1_model,
        tr,
        sig,
        p0=[s0_init, t1_init, 0],
        maxfev=10000
    )

    s0, t1, c = popt
    s0_err, t1_err, c_err = np.sqrt(np.diag(pcov))

    fit_vals = t1_model(tr, s0, t1, c)
    residuals = sig - fit_vals
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((sig - np.mean(sig)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return s0, t1, c, s0_err, t1_err, r2


def main():
    t1_results_rows = []
    scan_signal_map = {}
    all_tr_values = set()

    for scan_no in SCAN_NUMBERS:
        dicom_folder = os.path.join(BASE_FOLDER, str(scan_no), "pdata", "1", "dicom")
        print(f"\n[SCAN {scan_no}] {dicom_folder}")

        images, repetition_times, ds0 = load_dicom_series(dicom_folder)
        img0 = images[0]

        pixel_spacing = getattr(ds0, "PixelSpacing", [1.0, 1.0])
        mask = create_circle_mask_mm(img0.shape, pixel_spacing, CIRCLE_CENTERS_MM, CIRCLE_RADIUS_MM)

        mean_signals = []
        tr_to_sig = {}

        for tr, img in zip(repetition_times, images):
            val = float(np.mean(img[mask]))
            mean_signals.append(val)
            tr_to_sig[float(tr)] = val
            all_tr_values.add(float(tr))

        scan_signal_map[scan_no] = tr_to_sig

        # Fit
        try:
            s0, t1, c, s0_err, t1_err, r2 = fit_t1_and_r2(repetition_times, mean_signals)
            print(f"T1 = {t1:.2f} ms | R² = {r2:.3f}")
        except Exception as e:
            print("FIT FAIL:", e)
            s0 = t1 = c = s0_err = t1_err = r2 = np.nan

        # ---------- Excel result row ----------
        t1_results_rows.append({
            "Scan_no": scan_no,
            "T1_ms": t1,
            "T1_error_ms": t1_err,
            "R2": r2,
        })

    # =====================================================
    # 🔥 LAG DATAPOINTS (DET DU MANGLER)
    # =====================================================

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

    df_results = pd.DataFrame(t1_results_rows)
    df_dp_raw = pd.DataFrame(dp_raw)
    df_dp_norm = pd.DataFrame(dp_norm)

    # =====================================================
    # 🔥 SKRIV TIL EXCEL (FLERE SHEETS)
    # =====================================================

    out_xlsx = os.path.join(BASE_FOLDER, OUTPUT_EXCEL)

    with pd.ExcelWriter(out_xlsx, engine="xlsxwriter") as writer:
        df_results.to_excel(writer, index=False, sheet_name="T1_Results")
        df_dp_raw.to_excel(writer, index=False, sheet_name="DataPoints")
        df_dp_norm.to_excel(writer, index=False, sheet_name="DataPoints_norm")

    print(f"\n[OUTPUT] Lagret: {out_xlsx}")


if __name__ == "__main__":
    main()