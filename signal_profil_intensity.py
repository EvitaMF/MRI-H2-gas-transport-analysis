import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import pydicom

plt.style.use("seaborn-v0_8-whitegrid")

# -------------------------------------------------
# PATH
# -------------------------------------------------

#DICOM_PATH = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\CH4_Characterization2\8\pdata\1\dicom\MRIm001.dcm"
DICOM_PATH = r"C:\Users\evita\OneDrive - University of Bergen\MRI_H2\diffusion_study\2026 02 Diffusion\MRI Scan data\CH4 N2 120 diff45V\20260209_130301_CH4_CH4_N2_120_diff45V_1_7\262\pdata\1\dicom\MRIm08.dcm"
# -------------------------------------------------
# LES DICOM
# -------------------------------------------------

ds = pydicom.dcmread(DICOM_PATH)
img = ds.pixel_array.astype(float)

if img.ndim == 3:
    img = img.mean(axis=2)

H, W = img.shape

print("Image shape:", H, "x", W)

# -------------------------------------------------
# GLOBAL SIGNAL RANGE (for fixed y-axis)
# -------------------------------------------------

signal_min = img.min()
signal_max = img.max()

# -------------------------------------------------
# PIXEL SIZE
# -------------------------------------------------

ps = ds.get("PixelSpacing")

if ps is not None:
    pixel_mm = float(ps[0])
else:
    pixel_mm = 1.0

width_mm = W * pixel_mm
height_mm = H * pixel_mm

extent = (0, width_mm, height_mm, 0)

# -------------------------------------------------
# FIGUR
# -------------------------------------------------

fig = plt.figure(figsize=(11,8))

ax_img = plt.axes([0.05,0.25,0.5,0.7])
ax_v = plt.axes([0.62,0.55,0.33,0.38])
ax_h = plt.axes([0.62,0.10,0.33,0.38])

ax_sx = plt.axes([0.05,0.18,0.5,0.03])
ax_sy = plt.axes([0.05,0.12,0.5,0.03])

im = ax_img.imshow(img, cmap="gray", origin="upper", extent=extent)

ax_img.set_xlabel("X (mm)")
ax_img.set_ylabel("Y (mm)")
ax_img.set_title("MRI signal intensity")

# -------------------------------------------------
# SLIDERS
# -------------------------------------------------

init_x = width_mm/2
init_y = height_mm/2

slider_x = Slider(ax_sx,"X (mm)",0,width_mm,valinit=init_x)
slider_y = Slider(ax_sy,"Y (mm)",0,height_mm,valinit=init_y)

vline = ax_img.axvline(init_x,color="magenta",lw=2)
hline = ax_img.axhline(init_y,color="cyan",lw=2)

# -------------------------------------------------
# KONVERTER POSISJON
# -------------------------------------------------

def mm_to_col(x):
    return int((x/width_mm)*(W-1))

def mm_to_row(y):
    return int((y/height_mm)*(H-1))

# -------------------------------------------------
# PROFILER (PIXEL FOR PIXEL)
# -------------------------------------------------

def vertical_profile(x):

    c = mm_to_col(x)

    prof = img[:, c]   # pixel intensitet

    y = np.arange(H) * pixel_mm

    return y, prof


def horizontal_profile(y):

    r = mm_to_row(y)

    prof = img[r, :]   # pixel intensitet

    x = np.arange(W) * pixel_mm

    return x, prof

# -------------------------------------------------
# UPDATE
# -------------------------------------------------

def update(val=None):

    xm = slider_x.val
    ym = slider_y.val

    vline.set_xdata([xm])
    hline.set_ydata([ym])

    # -------- vertical profile --------

    y, pv = vertical_profile(xm)

    ax_v.cla()

    ax_v.plot(y, pv, linewidth=1.5)

    ax_v.set_xlabel("Y (mm)")
    ax_v.set_ylabel("Signal intensity")

    ax_v.set_ylim(signal_min, signal_max)  # låser y-aksen

    ax_v.legend()
    ax_v.grid()

    # -------- horizontal profile --------

    x, ph = horizontal_profile(ym)

    ax_h.cla()

    ax_h.plot(x, ph, linewidth=1.5)

    ax_h.set_xlabel("X (mm)")
    ax_h.set_ylabel("Signal intensity")

    ax_h.set_ylim(signal_min, signal_max)  # låser y-aksen

    ax_h.legend()
    ax_h.grid()

    fig.canvas.draw_idle()


slider_x.on_changed(update)
slider_y.on_changed(update)

update()

plt.show()