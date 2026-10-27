"""
Wire Rope Splice & Knot Detector
================================
This module processes MFL signals from wire ropes to automatically detect,
filter, and classify splices (knots) and repair sections. It creates a single
combined image containing a full-signal overview and zoomed-in subplots for all regions.
"""

import os
import glob
import string
import numpy as np
import scipy.signal as sp
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


class SpliceDetector:
    """
    A class to load MFL signals, detect splice regions using dynamic peak finding,
    classify the type of splice based on internal nodes, and generate combined visual reports.
    """

    def __init__(
        self,
        multiplier: float = 8.0,
        gap_threshold_m: float = 40.0,
        zoom_padding_m: float = 5.0,
    ):
        """
        Initializes the SpliceDetector with signal processing parameters.

        Parameters
        ----------
            multiplier (float): Factor to determine the dynamic prominence of peaks.
            gap_threshold_m (float): Maximum gap in meters to consider peaks as part of the same region.
            zoom_padding_m (float): Padding in meters to add to the left and right of a zoomed region plot.
        """
        self.multiplier = multiplier
        self.gap_threshold_m = gap_threshold_m
        self.zoom_padding_m = zoom_padding_m

    @staticmethod
    def load_signal(path: str) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Loads the .npz signal file and computes spatial coordinates.
        """
        with np.load(path, allow_pickle=False) as signal:
            sample_count = int(signal["sample_count"])
            x = np.linspace(
                float(signal["position_start_m"]),
                float(signal["position_end_m"]),
                sample_count,
            )
            y = signal["average_numerator"].astype(np.float64) / signal[
                "average_denominator"
            ].astype(np.float64)
            sample_to_meter = (
                float(signal["position_end_m"]) - float(signal["position_start_m"])
            ) / sample_count

        return x, y, sample_to_meter

    def _extract_peaks_and_regions(self, y: np.ndarray, stm: float) -> list[list[int]]:
        """
        Identifies prominent peaks and groups them into proximity-based regions.
        """
        median_val = np.median(y)
        abs_centered_y = np.abs(y - median_val)
        dynamic_prominence = np.mean(abs_centered_y) * self.multiplier
        dist = int(0.1 // stm)

        peaks, _ = sp.find_peaks(
            abs_centered_y,
            height=max(abs_centered_y) * 0.25,
            prominence=dynamic_prominence,
            distance=dist,
        )

        peaks = peaks[(y - median_val)[peaks] > 0]

        regions = []
        if len(peaks) == 0:
            return regions

        group_gap_threshold = int(self.gap_threshold_m // stm)
        current_region = [peaks[0]]

        for p in peaks[1:]:
            if p - current_region[-1] <= group_gap_threshold:
                current_region.append(p)
            else:
                regions.append(current_region)
                current_region = [p]
        regions.append(current_region)

        return regions

    def _filter_regions(self, regions: list[list[int]]) -> list[list[int]]:
        """
        Filters out spurious peaks within regions based on distance thresholds.
        """
        valid_regions = []
        regionss = [list(region) for region in regions if len(region) >= 2]

        for reg in regionss:
            limit = np.median(np.diff(reg)) // 1.06

            while len(reg) >= 2:
                if (reg[1] - reg[0]) < limit:
                    del reg[1]
                if reg[-1] - reg[-2] < limit:
                    del reg[-2]
                else:
                    break

            if len(reg) < 2:
                continue

            ref_dist = min(reg[1] - reg[0], reg[-1] - reg[-2])
            t = ref_dist / 1.8

            j = 2
            while j < len(reg) - 1:
                dist = reg[j + 1] - reg[j]
                if dist < t:
                    del reg[j + 1]
                else:
                    j += 2

            if len(reg) % 2 != 0:
                reg.pop()

            if len(reg) >= 2:
                valid_regions.append(reg)

        return valid_regions

    def _analyze_region_nodes(
        self, x: np.ndarray, y: np.ndarray, region: list[int]
    ) -> tuple[list[int], str, float]:
        """
        Finds inner nodes (valleys) of a region and classifies the splice type.
        """
        nodes = []
        for j in range(0, len(region) - 1, 2):
            p1, p2 = region[j], region[j + 1]
            L = p2 - p1
            search_start = p1 + int(0.1 * L)
            search_end = p1 + int(0.9 * L)

            if search_start >= search_end:
                node_idx = (p1 + p2) // 2
            else:
                segment = y[search_start:search_end]
                node_idx = search_start + np.argmin(segment)
            nodes.append(node_idx)

        node_count = len(nodes)

        if node_count == 6:
            splice_type = "Standard Splice"
        elif node_count in [7, 8]:
            splice_type = "Repair Splice"
        else:
            splice_type = "Local Repair"

        total_len = x[region[-1]] - x[region[0]]

        return nodes, splice_type, total_len

    def process_file(
        self, file_path: str, save_dir: str = None, show_plot: bool = True
    ):
        """
        Processes a single .npz file, analyzes regions, and generates a combined plot
        (Full View on top, Zoomed regions below with node distances).

        Parameters
        ----------
            file_path (str): The path to the .npz file.
            save_dir (str, optional): Directory to save the combined image.
                                      If None, plots are displayed dynamically instead of saved.
            show_plot (bool): If True, displays the combined plot at the end.
        """
        file_name = os.path.basename(file_path).replace(".npz", "")
        x, y, stm = self.load_signal(file_path)
        all_regions = self._extract_peaks_and_regions(y, stm)
        valid_regions = self._filter_regions(all_regions)

        letters = list(string.ascii_uppercase)
        num_regions = len(valid_regions)

        if num_regions == 0:
            print(f"No valid splice regions detected in {file_name}.")
            return

        total_plots = num_regions + 1
        fig, axes = plt.subplots(
            total_plots, 1, figsize=(14, 5.5 * total_plots), squeeze=False
        )
        axes = axes.flatten()

        ax_full = axes[0]
        ax_full.plot(x, y, color="black", linewidth=1, alpha=0.6, label="Signal")
        ax_full.set_title(
            f"MFL Signal Full View - {file_name}", fontsize=16, fontweight="bold"
        )
        ax_full.set_xlabel("Position (m)", fontsize=11)
        ax_full.set_ylabel("Amplitude", fontsize=11)

        for i, region in enumerate(valid_regions):
            nodes, _, _ = self._analyze_region_nodes(x, y, region)

            ax_full.plot(
                [x[region[0]], x[region[-1]]],
                [y[region[0]], y[region[-1]]],
                "ro",
                markersize=6,
                label="Start/End Peaks" if i == 0 else "",
            )

            if nodes:
                ax_full.plot(
                    x[nodes],
                    y[nodes],
                    "bo",
                    markersize=6,
                    label="Splice Nodes" if i == 0 else "",
                )

        handles, labels = ax_full.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_full.legend(by_label.values(), by_label.keys(), loc="upper right")

        for i, region in enumerate(valid_regions):
            nodes, splice_type, total_len = self._analyze_region_nodes(x, y, region)
            ax = axes[i + 1]

            ax.plot(x, y, color="black", linewidth=1.5, alpha=0.7, label="Signal")

            start_x_val = x[region[0]]
            end_x_val = x[region[-1]]
            start_y_val = y[region[0]]
            end_y_val = y[region[-1]]

            ax.plot(
                [start_x_val, end_x_val],
                [start_y_val, end_y_val],
                "ro",
                markersize=7,
                label="Start/End Peaks",
            )

            ax.text(
                start_x_val,
                start_y_val,
                "  Start",
                color="red",
                fontsize=12,
                fontweight="bold",
                verticalalignment="bottom",
            )
            ax.text(
                end_x_val,
                end_y_val,
                "  End",
                color="red",
                fontsize=12,
                fontweight="bold",
                verticalalignment="bottom",
            )

            start_x = start_x_val - self.zoom_padding_m
            end_x = end_x_val + self.zoom_padding_m
            ax.set_xlim(start_x, end_x)

            if len(nodes) > 0:
                ax.plot(x[nodes], y[nodes], "bo", markersize=7, label="Splice Nodes")
                for k, node_idx in enumerate(nodes):
                    ax.text(
                        x[node_idx],
                        y[node_idx],
                        f"  {letters[k]}",
                        color="blue",
                        fontsize=13,
                        fontweight="bold",
                        verticalalignment="bottom",
                    )

            ax.set_title(
                f"Region {i+1} | Type: {splice_type} | Nodes: {len(nodes)}",
                fontsize=14,
                fontweight="bold",
            )
            ax.set_ylabel("Amplitude", fontsize=11)
            ax.legend(loc="upper right")

            region_str = f"Length = {total_len:.2f}m\n"
            dist_strs = []

            if len(nodes) > 0:
                dist_start = x[nodes[0]] - start_x_val
                dist_strs.append(f"Start \u2192 {letters[0]} = {dist_start:.2f}m")
                for k in range(len(nodes) - 1):
                    dist = x[nodes[k + 1]] - x[nodes[k]]
                    dist_strs.append(
                        f"{letters[k]} \u2192 {letters[k+1]} = {dist:.2f}m"
                    )
                dist_end = end_x_val - x[nodes[-1]]
                dist_strs.append(
                    f"{letters[len(nodes)-1]} \u2192 End = {dist_end:.2f}m"
                )
                region_str += " | ".join(dist_strs)
            else:
                region_str += "No internal nodes found."

            ax.set_xlabel(
                f"Position (m)\n\n{region_str}", fontsize=12, fontweight="bold"
            )

        plt.tight_layout(h_pad=4.0)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            full_save_path = os.path.join(save_dir, f"{file_name}_Splice_Report.png")
            plt.savefig(full_save_path, dpi=200, bbox_inches="tight")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def process_directory(self, input_dir: str, save_dir: str = None):
        """
        Batch processes all .npz files in a given directory and generates combined plots.

        Parameters
        ----------
            input_dir (str): Directory containing the raw .npz files.
            save_dir (str, optional): Directory where the output image will be saved.
        """
        if save_dir and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_paths = glob.glob(os.path.join(input_dir, "*.npz"))

        for file_path in file_paths:
            show_display = False if save_dir else True
            self.process_file(file_path, save_dir=save_dir, show_plot=show_display)
