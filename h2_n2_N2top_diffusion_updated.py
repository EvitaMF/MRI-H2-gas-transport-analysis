
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os

# -------------------------------------------------
# Plot style
# -------------------------------------------------
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300
})
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False

figsize_small = (5, 3)

# ============================================
# 1. USER SETTINGS
# ============================================

excel_path = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\diffusion_study\2026 02 Diffusion\MRI Scan data\H2 N2 110 diff45 N2top\H2_N2_diffusion_N2top_results.xlsx"

DATASET_LABEL = "H$_2$–N$_2$ (N$_2$ top, H$_2$ bottom)"

SELECTED_SLICE = 11

# Velg signal:
# "N2_top" = ROI1_top_N2_mean
# "H2_bottom" = ROI2_bottom_H2_mean
FIT_SIGNAL = "N2_top"

# Fittevindu i timer
FIT_START_H = 0.0
FIT_END_H = 19.0

# Plottevindu
PLOT_START_H = 0.0
PLOT_END_H = None   # None = vis til siste datapunkt

# Sett til False mens du tester, så du ikke lagrer mange feil-rader i Excel
SAVE_RESULTS = True

# ============================================
# 2. PHYSICAL PARAMETERS
# ============================================

V_A = 7.06858e-5      # m^3
V_B = 7.06858e-5      # m^3

A_tube = 2.82743e-5   # m^2
L_tube = 0.201        # m

# ============================================
# 3. SIGNAL CONFIGURATION
# ============================================

signal_config = {
    "N2_top": {
        "mean_col": "ROI1_top_N2_mean",
        "std_col": "ROI1_top_N2_std",
        "label": "N$_2$ top chamber",
        "color": "tab:blue",
        "save_name": "N2_top",
        "chamber": "N2 top chamber"
    },
    "H2_bottom": {
        "mean_col": "ROI2_bottom_H2_mean",
        "std_col": "ROI2_bottom_H2_std",
        "label": "H$_2$ bottom chamber",
        "color": "tab:green",
        "save_name": "H2_bottom",
        "chamber": "H2 bottom chamber"
    }
}

if FIT_SIGNAL not in signal_config:
    raise ValueError("FIT_SIGNAL must be either 'N2_top' or 'H2_bottom'.")

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

required_cols = ["Scan", "Time", "Slice", mean_col, std_col]
missing_cols = [col for col in required_cols if col not in df.columns]

if missing_cols:
    raise ValueError(f"Missing columns in Excel file: {missing_cols}")

# Velg slice
df = df[df["Slice"] == SELECTED_SLICE].copy()

if df.empty:
    raise ValueError(f"No data found for Slice = {SELECTED_SLICE}")

# Sorter først etter scan
df = df.sort_values("Scan").reset_index(drop=True)

# ============================================
# 5. RECALCULATE TIME FROM CLOCK TIME
# ============================================
# Viktig:
# Legger kun til 24 timer hvis klokken går kraftig bakover,
# f.eks. fra 23:59 til 00:01.
# Små hopp bakover, f.eks. 16:19 til 16:17, behandles IKKE som midnatt.

time_as_string = df["Time"].astype(str).str.strip()

# Dersom Excel leser tiden som "1900-01-01 16:06:05", hent bare klokkeslettdelen
time_as_string = time_as_string.str[-8:]

df["Time_dt"] = pd.to_datetime(
    time_as_string,
    format="%H:%M:%S",
    errors="coerce"
)

if df["Time_dt"].isna().any():
    bad_times = df.loc[df["Time_dt"].isna(), "Time"].unique()
    raise ValueError(f"Could not parse these Time values: {bad_times}")

clock_seconds = (
    df["Time_dt"].dt.hour * 3600
    + df["Time_dt"].dt.minute * 60
    + df["Time_dt"].dt.second
).to_numpy()

corrected_seconds = []
day_offset = 0
previous_time = None

for current_time in clock_seconds:
    if previous_time is not None:
        time_drop = previous_time - current_time

        # Kun faktisk midnatt-overgang, ikke små uregelmessigheter
        if current_time < previous_time and time_drop > 12 * 3600:
            day_offset += 86400

    corrected_seconds.append(current_time + day_offset)
    previous_time = current_time

df["Time_seconds_recalculated"] = (
    np.array(corrected_seconds) - corrected_seconds[0]
)

df["Time_hours_recalculated"] = df["Time_seconds_recalculated"] / 3600

# Sorter etter den nye tidsaksen
df = df.sort_values("Time_seconds_recalculated").reset_index(drop=True)

t = df["Time_seconds_recalculated"].to_numpy()
t_h = df["Time_hours_recalculated"].to_numpy()

S = df[mean_col].to_numpy()
S_std = df[std_col].to_numpy()

print("\n==============================")
print("TIME CHECK")
print("==============================")
print(f"First clock time: {df['Time'].iloc[0]}")
print(f"Last clock time:  {df['Time'].iloc[-1]}")
print(f"Total duration:   {t_h.max():.2f} h")
print(f"Number of points: {len(df)}")

# ============================================
# 6. SELECT FIT WINDOW
# ============================================

fit_start_s = FIT_START_H * 3600
fit_end_s = FIT_END_H * 3600

mask_fit = (t >= fit_start_s) & (t <= fit_end_s)

t_fit_abs = t[mask_fit]
S_fit_data = S[mask_fit]
S_std_fit_data = S_std[mask_fit]

if len(t_fit_abs) < 4:
    raise ValueError("Too few data points in the selected fitting window.")

# Setter fit-tiden til 0 ved første datapunkt i fittevinduet
t_fit_rel = t_fit_abs - t_fit_abs[0]

print("\n==============================")
print("FIT SETUP")
print("==============================")
print(f"Dataset: {DATASET_LABEL}")
print(f"Selected slice: {SELECTED_SLICE}")
print(f"Fitted chamber: {fitted_chamber}")
print(f"Fitted signal column: {mean_col}")
print(f"Fit window: {FIT_START_H:.2f}–{FIT_END_H:.2f} h")
print(f"Using {len(t_fit_rel)} points")

# ============================================
# 7. EXPONENTIAL MODEL
# ============================================

def exp_model(t, S_eq, S0, tau):
    return S_eq + (S0 - S_eq) * np.exp(-t / tau)

S_eq_guess = S_fit_data[-1]
S0_guess = S_fit_data[0]
tau_guess = max((t_fit_rel[-1] - t_fit_rel[0]) / 5, 1)

use_sigma = np.all(np.isfinite(S_std_fit_data)) and np.all(S_std_fit_data > 0)

if use_sigma:
    popt, pcov = curve_fit(
        exp_model,
        t_fit_rel,
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
        t_fit_rel,
        S_fit_data,
        p0=[S_eq_guess, S0_guess, tau_guess],
        bounds=([-np.inf, -np.inf, 0], [np.inf, np.inf, np.inf]),
        maxfev=10000
    )

S_eq_fit, S0_fit, tau_fit = popt
tau_std = np.sqrt(np.diag(pcov))[2]

# ============================================
# 8. DIFFUSION COEFFICIENT
# ============================================

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
# 9. PLOT
# ============================================

output_folder = os.path.dirname(excel_path)

# Fit curve only over fitting window
t_curve_rel = np.linspace(t_fit_rel.min(), t_fit_rel.max(), 500)
S_curve = exp_model(t_curve_rel, S_eq_fit, S0_fit, tau_fit)

# Convert fit curve back to absolute plot time in hours
t_curve_abs_h = (t_curve_rel + t_fit_abs[0]) / 3600

if PLOT_END_H is None:
    PLOT_END_H = t_h.max()

mask_plot = (t_h >= PLOT_START_H) & (t_h <= PLOT_END_H)

plt.figure(figsize=figsize_small)

plt.errorbar(
    t_h[mask_plot],
    S[mask_plot],
    yerr=S_std[mask_plot],
    color=signal_color,
    marker="o",
    markersize=2,
    linestyle="--",
    linewidth=0.8,
    capsize=2,
    label=signal_label
)

plt.plot(
    t_curve_abs_h,
    S_curve,
    color="black",
    linewidth=1.3,
    label="Exponential fit"
)

plt.axvspan(
    FIT_START_H,
    FIT_END_H,
    color="lightgray",
    alpha=0.20,
    zorder=-1
)

plt.xlim(PLOT_START_H, PLOT_END_H)

ymin = np.nanmin(S[mask_plot])
ymax = np.nanmax(S[mask_plot])
margin = 0.08 * (ymax - ymin)

if margin == 0:
    margin = 1

plt.ylim(ymin - margin, ymax + margin)

plt.xlabel("Time [h]")
plt.ylabel("Signal intensity")
plt.grid(alpha=0.3)
plt.legend(frameon=False, loc="best")
plt.tight_layout()

save_path = os.path.join(
    output_folder,
    f"diffusion_fit_{save_name}_slice{SELECTED_SLICE}_corrected_time.png"
)

plt.savefig(save_path)
plt.show()

print("\nPlot saved to:")
print(save_path)

# ============================================
# 10. SAVE RESULTS TO EXCEL
# ============================================

if SAVE_RESULTS:
    results_dict = {
        "Dataset": DATASET_LABEL,
        "Slice": SELECTED_SLICE,
        "Fit_signal_key": FIT_SIGNAL,
        "Fitted_chamber": fitted_chamber,
        "Fitted_signal_column": mean_col,
        "Time_source": "Time column recalculated, midnight threshold 12 h",
        "Total_duration_h": t_h.max(),
        "Fit_start_h": FIT_START_H,
        "Fit_end_h": FIT_END_H,
        "S_eq": S_eq_fit,
        "S0": S0_fit,
        "Tau_s": tau_fit,
        "Tau_std_s": tau_std,
        "D_m2_per_s": D,
        "D_std_m2_per_s": D_std,
        "D_cm2_per_s": D * 1e4,
        "D_std_cm2_per_s": D_std * 1e4,
        "D_formatted_cm2_per_s": f"{D*1e4:.3f} ± {D_std*1e4:.3f}",
        "N_points_used": len(t_fit_rel)
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

    print("\nResults updated in Excel.")
else:
    print("\nSAVE_RESULTS = False, so results were not written to Excel.")