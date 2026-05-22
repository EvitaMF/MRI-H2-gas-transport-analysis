import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# -------------------------------------------------
# Plot style
# -------------------------------------------------
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 12,
    "legend.fontsize": 9,
    "figure.dpi": 300
})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

figsize_small = (5, 3)

# ============================================
# 1. SELECT DATASET
# ============================================

dataset = {
    "path": r"C:\Users\efi015\OneDrive - University of Bergen\MRI_H2\diffusion_study\2026 02 Diffusion\MRI Scan data\H2 N2 110 diff45 horisontal\H2 N2 sidelengs\H2_N2_horisontal.xlsx",
    "cutoff": 13208,
    "label": "H$_2$–N$_2$ (H$_2$ left, N$_2$ right)"
}

# dataset = {
#     "path": r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\diffusion_study\2026 02 Diffusion\MRI Scan data\CH4 N2 120 diff45V\20260209_130301_CH4_CH4_N2_120_diff45V_1_7\CH4_N2_diff45v_diffusion_results.xlsx",
#     "cutoff": 78660,
#     "label": "CH$_4$–N$_2$ (120 bar)"
# }

excel_path = dataset["path"]
TIME_CUTOFF = dataset["cutoff"]
DATASET_LABEL = dataset["label"]

# Velg hvilken kurve som skal fittes:
# "H2" = ROI1_mean
# "N2" = ROI2_mean
# Endre kolonnenavnene i signal_config dersom Excel-arket ditt bruker andre ROI-navn.
FIT_SIGNAL = "H2"

# Hvis arket har en Slice-kolonne og du vil filtrere på slice, sett f.eks. SELECTED_SLICE = 11.
# Hvis ikke, behold None.
SELECTED_SLICE = None

SAVE_RESULTS = True

# ============================================
# 2. PHYSICAL PARAMETERS
# ============================================

V_A = 7.06858e-5   # m^3
V_B = 7.06858e-5   # m^3

A_tube = 2.82743e-5   # m^2
L_tube = 0.201        # m

# ============================================
# 3. SIGNAL CONFIGURATION
# ============================================

signal_config = {
    "H2": {
        "mean_col": "ROI1_left_H2_mean",
        "std_col": None,
        "label": "H$_2$ chamber",
        "color": "tab:green",
        "save_name": "H2",
        "chamber": "H2 chamber"
    },
    "N2": {
        "mean_col": "ROI2_right_N2_mean",
        "std_col": None,
        "label": "N$_2$ chamber",
        "color": "tab:blue",
        "save_name": "N2",
        "chamber": "N2 chamber"
    }
}

if FIT_SIGNAL not in signal_config:
    raise ValueError("FIT_SIGNAL must be one of: " + ", ".join(signal_config.keys()))

mean_col = signal_config[FIT_SIGNAL]["mean_col"]
std_col = signal_config[FIT_SIGNAL]["std_col"]
signal_label = signal_config[FIT_SIGNAL]["label"]
signal_color = signal_config[FIT_SIGNAL]["color"]
save_name = signal_config[FIT_SIGNAL]["save_name"]
fitted_chamber = signal_config[FIT_SIGNAL]["chamber"]

# ============================================
# 4. READ DATA
# ============================================

df = pd.read_excel(excel_path)

required_cols = ["Time_seconds", mean_col]
if SELECTED_SLICE is not None:
    required_cols.append("Slice")

missing_cols = [col for col in required_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f"Missing columns in Excel file: {missing_cols}")

if SELECTED_SLICE is not None:
    df = df[df["Slice"] == SELECTED_SLICE].copy()
    if df.empty:
        raise ValueError(f"No data found for Slice = {SELECTED_SLICE}")

df = df.sort_values("Time_seconds").reset_index(drop=True)

# Sørg for at tiden starter på 0 for valgt datautvalg
df["Time_seconds"] = df["Time_seconds"] - df["Time_seconds"].min()

t = df["Time_seconds"].to_numpy()
S = df[mean_col].to_numpy()

# Optional standard deviation column, if available
if std_col is not None and std_col in df.columns:
    S_std = df[std_col].to_numpy()
else:
    S_std = None

# ============================================
# 5. FILTER DATA
# ============================================

mask_fit = t <= TIME_CUTOFF

t_fit_data = t[mask_fit]
S_fit_data = S[mask_fit]
S_std_fit_data = S_std[mask_fit] if S_std is not None else None

if len(t_fit_data) < 4:
    raise ValueError("Too few data points for fitting. Increase TIME_CUTOFF or check selected data.")

print(f"\nDataset: {DATASET_LABEL}")
print(f"Fitted chamber: {fitted_chamber}")
print(f"Fitted signal column: {mean_col}")
if SELECTED_SLICE is not None:
    print(f"Selected slice: {SELECTED_SLICE}")
print(f"Using {len(t_fit_data)} points (t <= {TIME_CUTOFF} s)")

# ============================================
# 6. EXPONENTIAL MODEL
# ============================================

def exp_model(t, S_eq, S0, tau):
    return S_eq + (S0 - S_eq) * np.exp(-t / tau)

S_eq_guess = S_fit_data[-1]
S0_guess = S_fit_data[0]
tau_guess = max((t_fit_data[-1] - t_fit_data[0]) / 5, 1)

use_sigma = (
    S_std_fit_data is not None
    and np.all(np.isfinite(S_std_fit_data))
    and np.all(S_std_fit_data > 0)
)

if use_sigma:
    popt, pcov = curve_fit(
        exp_model,
        t_fit_data,
        S_fit_data,
        p0=[S_eq_guess, S0_guess, tau_guess],
        sigma=S_std_fit_data,
        absolute_sigma=False,
        bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
        maxfev=10000
    )
else:
    popt, pcov = curve_fit(
        exp_model,
        t_fit_data,
        S_fit_data,
        p0=[S_eq_guess, S0_guess, tau_guess],
        bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
        maxfev=10000
    )

S_eq_fit, S0_fit, tau_fit = popt

# ============================================
# 7. UNCERTAINTY AND DIFFUSION COEFFICIENT
# ============================================

tau_std = np.sqrt(np.diag(pcov))[2]

k = 1 / (tau_fit * (1 / V_A + 1 / V_B))
D = k * L_tube / A_tube
D_std = D * (tau_std / tau_fit)

print("\n==============================")
print("FIT RESULTS")
print("==============================")
print(f"S_eq = {S_eq_fit:.4f}")
print(f"S0   = {S0_fit:.4f}")
print(f"Tau  = {tau_fit:.4f} ± {tau_std:.4f} s")
print(f"D    = {D:.6e} ± {D_std:.2e} m^2/s")
print(f"D    = {D*1e4:.4f} ± {D_std*1e4:.4f} cm^2/s")

# ============================================
# 8. PLOT
# ============================================

output_folder = os.path.dirname(excel_path)

t_hours = t / 3600
t_fit_data_hours = t_fit_data / 3600
cutoff_hours = TIME_CUTOFF / 3600

t_fit_full = np.linspace(t_fit_data.min(), t_fit_data.max(), 500)
t_fit_hours = t_fit_full / 3600
S_fit = exp_model(t_fit_full, S_eq_fit, S0_fit, tau_fit)

mask_unused = t > TIME_CUTOFF

plt.figure(figsize=figsize_small)

if S_std_fit_data is not None:
    plt.errorbar(
        t_fit_data_hours,
        S_fit_data,
        yerr=S_std_fit_data,
        color=signal_color,
        marker="o",
        markersize=3,
        linestyle="--",
        linewidth=0.8,
        capsize=2,
        label=signal_label
    )
else:
    plt.plot(
        t_fit_data_hours,
        S_fit_data,
        "o--",
        color=signal_color,
        markersize=3,
        linewidth=0.8,
        label=signal_label
    )

if np.any(mask_unused):
    plt.plot(
        t_hours[mask_unused],
        S[mask_unused],
        "o",
        color=signal_color,
        alpha=0.15,
        markersize=3,
        label="Excluded data"
    )

plt.plot(
    t_fit_hours,
    S_fit,
    color="black",
    linewidth=1.2,
    label="Exponential fit"
)

plt.xlim(0, t_hours.max() * 1.05)

if TIME_CUTOFF < t.max():
    plt.axvspan(
        cutoff_hours,
        t_hours.max(),
        color="lightgray",
        alpha=0.35,
        zorder=-1
    )

ymin_all = np.nanmin(S)
ymax_all = np.nanmax(S)
margin = 0.06 * (ymax_all - ymin_all)
if margin == 0:
    margin = 1

plt.ylim(ymin_all - margin, ymax_all + margin)
plt.xlabel("Time [h]")
plt.ylabel("Signal intensity")
plt.grid(alpha=0.3)
plt.legend(frameon=False)
plt.tight_layout()

save_path = os.path.join(output_folder, f"diffusion_fit_{save_name}_clean.png")
plt.savefig(save_path)
plt.show()

print("Plot saved to:", save_path)

# ============================================
# 9. SAVE TO EXCEL
# ============================================

if SAVE_RESULTS:
    results_dict = {
        "Dataset": DATASET_LABEL,
        "Slice": SELECTED_SLICE,
        "Fit_signal_key": FIT_SIGNAL,
        "Fitted_chamber": fitted_chamber,
        "Fitted_signal_column": mean_col,
        "Time_cutoff_s": TIME_CUTOFF,
        "S_eq": S_eq_fit,
        "S0": S0_fit,
        "Tau_s": tau_fit,
        "Tau_std_s": tau_std,
        "Tau_hours": tau_fit / 3600,
        "k_m3_per_s": k,
        "D_m2_per_s": D,
        "D_std_m2_per_s": D_std,
        "D_cm2_per_s": D * 1e4,
        "D_std_cm2_per_s": D_std * 1e4,
        "D_formatted_cm2_per_s": f"{D*1e4:.3f} ± {D_std*1e4:.3f}",
        "N_points_used": len(t_fit_data)
    }

    results_df = pd.DataFrame([results_dict])
    sheet_name = "Fit_results"

    try:
        existing_df = pd.read_excel(excel_path, sheet_name=sheet_name)
        results_df = pd.concat([existing_df, results_df], ignore_index=True)
    except ValueError:
        pass

    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        results_df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("Results updated in Excel (sheet: Fit_results).")
else:
    print("SAVE_RESULTS = False, so results were not written to Excel.")
