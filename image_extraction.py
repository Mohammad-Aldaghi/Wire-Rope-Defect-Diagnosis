"""
Image Signal Extraction Module
==============================
This module extracts MFL (Magnetic Flux Leakage) signal curves from visual plot images
using color masking, baseline correction, and spline smoothing.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline


def extract_signal_from_image(
    gap: str, fault_type: str, want_info: bool = False, want_plot: bool = False
) -> np.ndarray:
    """
    Extracts, corrects, and resamples an MFL signal curve from a given plot image.

    Parameters
    ----------
        gap (str): The gap size identifier (e.g., "1", "10", "50").
        fault_type (str): The type of defect/fault (e.g., "king", "inner", "outer").
        want_info (bool): Whether to print extraction details and peak amplitude. Defaults to False.
        want_plot (bool): Whether to plot the extracted signal. Defaults to False.

    Returns:
        np.ndarray: The processed 1D signal values mapped to real-world scale.
    """
    REAL_X_RANGE = (-200, 200)
    REAL_Y_VISUAL_RANGE = (-1, 1)

    img = cv2.imread(f"docs/images/GAN images/{fault_type} {gap}.png")
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    h, w = mask.shape
    x_pixels_raw = []
    y_pixels_raw = []

    for col in range(w):
        y_indices = np.where(mask[:, col] == 255)[0]
        if len(y_indices) > 0:
            y_pixels_raw.append(h - np.median(y_indices))
            x_pixels_raw.append(col)

    x_pixels_raw = np.array(x_pixels_raw)
    y_pixels_raw = np.array(y_pixels_raw)

    start_baseline = np.mean(y_pixels_raw[:10])
    end_baseline = np.mean(y_pixels_raw[-10:])
    overall_baseline = (start_baseline + end_baseline) / 2
    y_pixels_corrected = y_pixels_raw - overall_baseline

    spline = UnivariateSpline(x_pixels_raw, y_pixels_corrected, s=50)
    x_pixel_grid = np.linspace(0, w - 1, 1000)
    y_pixel_smooth = spline(x_pixel_grid)

    real_x = np.linspace(REAL_X_RANGE[0], REAL_X_RANGE[1], 1000)
    pixels_per_unit_y = h / (REAL_Y_VISUAL_RANGE[1] - REAL_Y_VISUAL_RANGE[0])
    real_y = y_pixel_smooth / pixels_per_unit_y

    if want_info:
        print(f"Extracted Signal Info:")
        print(f"X range: {real_x[0]:.1f} to {real_x[-1]:.1f} mm")
        print(f"Detected Peak Amplitude: {np.max(np.abs(real_y)):.4f}")

    if want_plot:
        plt.figure(figsize=(10, 5))
        plt.plot(real_x, real_y, color="blue", linewidth=2)
        plt.xlim(REAL_X_RANGE)
        plt.ylim(REAL_Y_VISUAL_RANGE)
        plt.xticks(np.arange(-200, 201, 50))
        plt.yticks(np.arange(-1.0, 1.1, 0.5))
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.show()

    return real_y
