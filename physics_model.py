import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit
from scipy.interpolate import PchipInterpolator
import os

sns.set_theme(style="darkgrid")


class PhysicsInformedMFL:
    """
    A class to fit real MFL sensor data, extract magnetic dipole parameters,
    and generate synthetic signals for varying gap sizes and multi-wire breaks.
    """

    def __init__(self, npz_path: str):
        """
        Initializes the model by loading real MRT sensor datasets.

        Parameters
        ----------
            npz_path (str): Path to the .npz dataset file containing signals and labels.
        """
        self.data = np.load(npz_path)
        self.signals = self.data["signals"]
        self.labels = self.data["labels"]
        self.x_real = np.linspace(-200, 200, 1000)
        self.params_db = {}
        self.interpolators = {}
        self._fit_all_data()

    @staticmethod
    def mfl_formula(x: np.ndarray, A: float, d: float, sigma: float) -> np.ndarray:
        """
        Calculates the theoretical MFL signal using a double-Gaussian derivative model.

        Parameters
        ----------
            x (np.ndarray): Spatial positions along the cable (mm).
            A (float): Amplitude scaling factor.
            d (float): Distance between magnetic poles (mm).
            sigma (float): Dispersion parameter representing lift-off attenuation.

        Returns:
            np.ndarray: Calculated MFL signal values.
        """
        norm_factor = np.exp(0.5)
        pole1 = (
            ((x - d / 2) / sigma)
            * np.exp(-((x - d / 2) ** 2) / (2 * sigma**2))
            * norm_factor
        )
        pole2 = (
            ((x + d / 2) / sigma)
            * np.exp(-((x + d / 2) ** 2) / (2 * sigma**2))
            * norm_factor
        )
        return A * (pole2 - pole1)

    def _fit_all_data(self):
        """Fits the physical model to all real signals in the dataset to build a parameter database."""
        sorted_indices = np.argsort(self.labels)
        labels_sorted = self.labels[sorted_indices]
        signals_sorted = self.signals[sorted_indices]

        for gap, y_real in zip(labels_sorted, signals_sorted):
            initial_guess = [0.5, gap, 10.0]
            valid_indices = np.abs(y_real) < 0.95
            x_valid = self.x_real[valid_indices]
            y_valid = y_real[valid_indices]

            try:
                lower_bounds = [0.001, max(0.0, gap - 5.0), 2.0]
                upper_bounds = [20.0, gap + 5.0, 14]
                param_bounds = (lower_bounds, upper_bounds)

                popt, _ = curve_fit(
                    self.mfl_formula,
                    x_valid,
                    y_valid,
                    p0=initial_guess,
                    bounds=param_bounds,
                    maxfev=10000,
                )

                peak_val = np.max(np.abs(y_valid))

                self.params_db[gap] = {
                    "A": popt[0],
                    "d": popt[1],
                    "sigma": popt[2],
                    "peak": peak_val,
                }
            except RuntimeError:
                pass

        self._build_interpolators()

    def _build_interpolators(self):
        """Builds PCHIP interpolators for smooth parameter generation across unmeasured gaps."""
        gaps = np.array(sorted(list(self.params_db.keys())))
        A_vals = np.array([self.params_db[g]["A"] for g in gaps])
        d_vals = np.array([self.params_db[g]["d"] for g in gaps])
        sigma_vals = np.array([self.params_db[g]["sigma"] for g in gaps])
        peak_vals = np.array([self.params_db[g]["peak"] for g in gaps])

        self.interpolators["A"] = PchipInterpolator(gaps, A_vals, extrapolate=True)
        self.interpolators["d"] = PchipInterpolator(gaps, d_vals, extrapolate=True)
        self.interpolators["sigma"] = PchipInterpolator(
            gaps, sigma_vals, extrapolate=True
        )
        self.interpolators["peak"] = PchipInterpolator(
            gaps, peak_vals, extrapolate=True
        )

    def generate_signal(
        self, target_gap: float, sample_size: int = 128
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generates a physics-informed MFL signal for an arbitrary gap size.

        Parameters
        ----------
            target_gap (float): The target wire break gap size in millimeters (mm).
            sample_size (int): Number of points for the output sliced window.

        Returns:
            tuple[np.ndarray, np.ndarray]: Sliced spatial positions (x) and generated signal values (y).
        """
        A_pred = float(self.interpolators["A"](target_gap))
        d_pred = float(self.interpolators["d"](target_gap))
        sigma_pred = float(self.interpolators["sigma"](target_gap))
        target_peak = float(self.interpolators["peak"](target_gap))

        y_full = self.mfl_formula(self.x_real, A_pred, d_pred, sigma_pred)

        current_peak = np.max(np.abs(y_full))
        if current_peak > 0:
            y_full = y_full * (target_peak / current_peak)

        center = len(y_full) // 2
        half_size = sample_size // 2
        start = center - half_size
        end = start + sample_size

        y_sliced = y_full[start:end]
        x_sliced = self.x_real[start:end]

        return x_sliced, y_sliced

    def plot_prediction(
        self, target_gap: float, compare_real_gaps=None, save_path: str = None
    ) -> None:
        """
        Plots the generated physics model prediction against real sensor data.
        Note: Real sensor dataset typically contains specific gap benchmarks: 1, 5, 10, 50, 100 mm.

        Parameters
        ----------
            target_gap (float): Gap size for the physics-generated model prediction.
            compare_real_gaps (float | list[float], optional): Real gap(s) to overlay for comparison
                                                                (e.g., 10 or [10, 50]). Defaults to None.
            save_path (str, optional): File path to save the generated plot image. Defaults to None.

        Returns:
            None
        """
        x, y_gen = self.generate_signal(target_gap, sample_size=len(self.x_real))

        plt.figure(figsize=(10, 5))
        title_text = f"Generated Signal (Blue) for {target_gap}mm"

        if compare_real_gaps is not None:
            if not isinstance(compare_real_gaps, (list, tuple, np.ndarray)):
                compare_real_gaps = [compare_real_gaps]

            title_text = f"Blue: Generated ({target_gap}mm) | Real Signals: {compare_real_gaps}mm"
            colors = ["red", "green", "orange", "purple", "brown", "cyan"]

            for i, gap in enumerate(compare_real_gaps):
                if gap in self.labels:
                    idx = np.where(self.labels == gap)[0][0]
                    color = colors[i % len(colors)]
                    plt.plot(
                        self.x_real,
                        self.signals[idx],
                        color=color,
                        linewidth=2.0,
                        alpha=0.8,
                        label=f"Real {gap}mm (OITAF Data)",
                    )
                else:
                    print(
                        f"Warning: Real gap {gap}mm not found in dataset (Available benchmarks: 1, 5, 10, 50, 100 mm)."
                    )

        plt.plot(
            x,
            y_gen,
            color="blue",
            linewidth=2.5,
            linestyle="-",
            zorder=10,
            label=f"Generated {target_gap}mm (Physics Model)",
        )

        # plt.title(title_text, fontsize=14, fontweight="bold")
        plt.xlabel("Position along rope (mm)")
        plt.ylabel("Magnetic Flux Leakage")
        plt.xlim(-200, 200)
        plt.ylim(-1, 1)
        plt.legend(loc="upper right")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, bbox_inches="tight", dpi=300)

        plt.show()

    def plot_parameters_trend(self, max_gap_plot: float = 150.0) -> None:
        """
        Plots how fitted MFL parameters (Amplitude, Pole Distance, Dispersion) change across different gap sizes.
        Includes an extrapolated trend line up to `max_gap_plot`.
        """
        gaps = np.array(sorted(list(self.params_db.keys())))
        A_vals = np.array([self.params_db[g]["A"] for g in gaps])
        d_vals = np.array([self.params_db[g]["d"] for g in gaps])
        sigma_vals = np.array([self.params_db[g]["sigma"] for g in gaps])

        trend_gaps = np.linspace(min(gaps), max(gaps.max(), max_gap_plot), 500)
        A_trend = self.interpolators["A"](trend_gaps)
        d_trend = self.interpolators["d"](trend_gaps)
        sigma_trend = self.interpolators["sigma"](trend_gaps)

        fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
        fig.suptitle(
            "MFL Physics Parameters vs. Wire Break Gap Size",
            fontsize=15,
            fontweight="bold",
        )

        axs[0].plot(trend_gaps, A_trend, color="lightblue", linestyle="-", linewidth=2)
        axs[0].plot(
            gaps, A_vals, color="blue", marker="o", linestyle="None", markersize=6
        )
        axs[0].set_ylabel("Amplitude (A)")
        axs[0].set_title("Magnetic Intensity Scaling")

        axs[1].plot(trend_gaps, d_trend, color="lightgreen", linestyle="-", linewidth=2)
        axs[1].plot(
            gaps,
            d_vals,
            color="green",
            marker="s",
            linestyle="None",
            markersize=6,
            label="Fitted Pole Distance (d)",
        )
        axs[1].plot(
            trend_gaps,
            trend_gaps,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="Ideal 1:1 Tracking (d = gap)",
        )
        axs[1].set_ylabel("Pole Distance (d) [mm]")
        axs[1].set_title("Physical Distance Between Magnetic Poles")
        axs[1].legend()

        axs[2].plot(
            trend_gaps, sigma_trend, color="thistle", linestyle="-", linewidth=2
        )
        axs[2].plot(
            gaps, sigma_vals, color="purple", marker="^", linestyle="None", markersize=6
        )
        axs[2].set_ylabel("Dispersion (\u03c3)")
        axs[2].set_xlabel("Target Gap Size (mm)")
        axs[2].set_title("Field Spread (Lift-off Attenuation)")

        axs[2].set_xlim(0, max_gap_plot)

        plt.tight_layout()
        plt.subplots_adjust(top=0.91)
        plt.show()

    def multi_wire_break(
        self,
        distance: float,
        num_breaks: int,
        gap: float = 10.0,
        fault_type: str = "outer",
        want_plot: bool = True,
        data_dir: str = "../data/Inner King Outer dataset",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simulates multi-wire breaks by dynamically summing individual shifted MFL signals.

        Parameters
        ----------
            distance (float): Spacing distance between consecutive wire breaks (mm).
            num_breaks (int): Total number of wire breaks.
            gap (float): Gap size for each individual break (mm). Defaults to 10.0.
            fault_type (str): Type of fault dataset to use ('inner', 'outer', 'king').
            want_plot (bool): Whether to plot the resulting multi-break signal.
            data_dir (str): Base directory path for the dataset files to handle relative pathing.

        Returns:
            tuple[np.ndarray, np.ndarray]: Common spatial axis (x_common) and total superimposed signal (y_total).
        """
        positions = [i * distance for i in range(num_breaks)]
        center_offset = (positions[-1] - positions[0]) / 2
        positions = [p - center_offset for p in positions]

        signals = []

        for d in positions:
            x, y = self.generate_signal(target_gap=gap, sample_size=len(self.x_real))
            signals.append((x + d, y))

        padding = 150
        x_min = positions[0] - padding
        x_max = positions[-1] + padding
        dx = 1
        x_common = np.arange(x_min, x_max + dx, dx)

        y_total = np.zeros_like(x_common, dtype=float)
        for x, y in signals:
            y_interp = np.interp(x_common, x, y, left=0, right=0)
            y_total += y_interp

        if want_plot:
            plt.figure(figsize=(12, 4.5))
            plt.plot(
                x_common,
                y_total,
                "r",
                linewidth=2.0,
                label=f"Sum of {num_breaks} breaks",
            )
            plt.title(
                f"Multi Wire Break ({num_breaks}x {fault_type.capitalize()} | Gap: {gap}mm | Dist: {distance}mm)",
                fontsize=13,
                fontweight="bold",
            )
            plt.xlabel("Position along rope (mm)")
            plt.ylabel("Magnetic Flux Leakage")
            plt.xlim(x_min, x_max)
            y_bound = max(np.max(np.abs(y_total)) * 1.2, 1.0)
            plt.ylim(-y_bound, y_bound)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return x_common, y_total
