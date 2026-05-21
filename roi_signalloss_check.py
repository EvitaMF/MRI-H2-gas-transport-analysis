import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# =====================================================
# CONFIGURATION
# =====================================================
2
EXCEL_PATH = r"C:\Users\efi015\OneDrive - University of Bergen\MRI_H2\diffusion_study\2026 02 Diffusion\MRI Scan data\H2 N2 110 diff45 horisontal\H2 N2 sidelengs\H2_N2_horisontal.xlsx"

SHEET_NAME = "Sheet1"   # endre hvis arket heter noe annet

# Kolonner i Excel-arket ditt
SCAN_COL = "Scan"
TIME_COL = "Time_minutes"     # bruk None hvis du heller vil plotte mot scan number

ROI1_COL = "ROI1_left_H2_mean"    # endre til faktisk kolonnenavn
ROI2_COL = "ROI2_right_N2_mean"    # endre til faktisk kolonnenavn

# Output
OUTPUT_NAME = "signal_loss_QC_from_existing_ROIs"


# =====================================================
# READ DATA
# =====================================================

df = pd.read_excel(EXCEL_PATH, sheet_name=SHEET_NAME)

# Sorter data
if TIME_COL is not None and TIME_COL in df.columns:
    df = df.sort_values(TIME_COL).reset_index(drop=True)
    x = df[TIME_COL]
    x_label = "Time [h]"
else:
    df = df.sort_values(SCAN_COL).reset_index(drop=True)
    x = df[SCAN_COL]
    x_label = "Scan number"


# =====================================================
# SIGNAL-LOSS QC
# =====================================================

# Total signal from both chambers
df["Total_signal"] = df[ROI1_COL] + df[ROI2_COL]

# Normalize to first total signal
S_total_0 = df["Total_signal"].iloc[0]

df["ROI1_percent_of_initial_total"] = df[ROI1_COL] / S_total_0 * 100
df["ROI2_percent_of_initial_total"] = df[ROI2_COL] / S_total_0 * 100
df["Total_signal_percent"] = df["Total_signal"] / S_total_0 * 100


# =====================================================
# SUMMARY
# =====================================================

mean_total = df["Total_signal_percent"].mean()
std_total = df["Total_signal_percent"].std()
min_total = df["Total_signal_percent"].min()
max_total = df["Total_signal_percent"].max()

print("Signal-loss QC summary")
print("----------------------")
print(f"Mean total signal: {mean_total:.2f} %")
print(f"Std total signal:  {std_total:.2f} %")
print(f"Min total signal:  {min_total:.2f} %")
print(f"Max total signal:  {max_total:.2f} %")


# =====================================================
# PLOT
# =====================================================

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 300,
})

fig, axes = plt.subplots(
    2, 1,
    figsize=(7, 6),
    sharex=True,
    gridspec_kw={"height_ratios": [1.2, 1]}
)

ax1, ax2 = axes

# -----------------------------------------------------
# Panel A: redistribution between chambers
# -----------------------------------------------------

ax1.plot(
    x,
    df["ROI1_percent_of_initial_total"],
    marker="o",
    markersize=3,
    linestyle="--",
    linewidth=0.9,
    label="ROI 1"
)

ax1.plot(
    x,
    df["ROI2_percent_of_initial_total"],
    marker="o",
    markersize=3,
    linestyle="--",
    linewidth=0.9,
    label="ROI 2"
)

ax1.set_ylabel("Signal / initial total signal [%]")
ax1.set_title("Signal redistribution between chambers")
ax1.grid(alpha=0.3)
ax1.legend(frameon=True)


# -----------------------------------------------------
# Panel B: summed signal
# -----------------------------------------------------

ax2.plot(
    x,
    df["Total_signal_percent"],
    marker="o",
    markersize=3,
    linestyle="--",
    linewidth=0.9,
    label="Summed signal"
)

ax2.axhline(
    100,
    linestyle="-",
    linewidth=1,
    label="Initial total signal"
)

ax2.axhspan(
    95,
    105,
    alpha=0.15,
    label="95–105 % range"
)

ax2.set_xlabel(x_label)
ax2.set_ylabel("Total signal [%]")
ax2.set_title("Signal-loss quality control")
ax2.grid(alpha=0.3)
ax2.legend(frameon=True)

for ax in axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

fig.tight_layout()


# =====================================================
# SAVE
# =====================================================

excel_path = Path(EXCEL_PATH)
output_folder = excel_path.parent

plot_path = output_folder / f"{OUTPUT_NAME}.png"
new_excel_path = output_folder / f"{OUTPUT_NAME}.xlsx"

fig.savefig(plot_path, dpi=300, bbox_inches="tight")

with pd.ExcelWriter(new_excel_path, engine="openpyxl") as writer:
    df.to_excel(writer, sheet_name="Signal_loss_QC", index=False)

    summary = pd.DataFrame({
        "Parameter": [
            "Mean total signal [%]",
            "Std total signal [%]",
            "Min total signal [%]",
            "Max total signal [%]",
            "Initial total signal [a.u.]",
        ],
        "Value": [
            mean_total,
            std_total,
            min_total,
            max_total,
            S_total_0,
        ]
    })

    summary.to_excel(writer, sheet_name="Summary", index=False)

print(f"\nSaved plot to:\n{plot_path}")
print(f"\nSaved Excel file to:\n{new_excel_path}")

plt.show()