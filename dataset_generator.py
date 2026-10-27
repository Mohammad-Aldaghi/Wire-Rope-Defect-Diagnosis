import os
import numpy as np
from src.physics_model import PhysicsInformedMFL


class DatasetGenerator:
    """
    Generates uniform synthetic datasets for Single and Multi wire breaks separately.
    Gap sizes are uniformly distributed between 1 and 100 mm.
    """

    def __init__(self, samples_per_class: int = 5000, window_size: int = 1000):
        self.samples_per_class = samples_per_class
        self.window_size = window_size
        self.fault_mapping = {"king": 0, "inner": 1, "outer": 2}

    def _pad_or_crop(self, signal: np.ndarray) -> np.ndarray:
        length = len(signal)
        if length == self.window_size:
            return signal
        elif length > self.window_size:
            start = (length - self.window_size) // 2
            return signal[start : start + self.window_size]
        else:
            pad_total = self.window_size - length
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            return np.pad(signal, (pad_left, pad_right), mode="constant")

    def generate_single_dataset(self, data_dir: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        X, y_class, y_gap = [], [], []

        for fault_type, class_label in self.fault_mapping.items():
            print(f"Generating Single Break data for: {fault_type}...")
            model = PhysicsInformedMFL(f"{data_dir}/{fault_type}_fault_dataset.npz")
            for _ in range(self.samples_per_class):
                gap = np.random.uniform(1.0, 100.0)
                _, y = model.generate_signal(
                    target_gap=gap, sample_size=self.window_size
                )

                X.append(self._pad_or_crop(y))
                y_class.append(class_label)
                y_gap.append(gap)

        save_path = os.path.join(save_dir, "single_dataset.npz")
        np.savez(
            save_path, X=np.array(X), y_class=np.array(y_class), y_gap=np.array(y_gap)
        )
        print(f"Single dataset saved to: {save_path}")

    def generate_multi_dataset(self, data_dir: str, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        X, y_class, y_gap, y_break_count, y_distance = [], [], [], [], []

        for fault_type, class_label in self.fault_mapping.items():
            print(f"Generating Multi Break data for: {fault_type}...")
            model = PhysicsInformedMFL(f"{data_dir}/{fault_type}_fault_dataset.npz")
            for _ in range(self.samples_per_class):
                gap = np.random.uniform(1.0, 100.0)
                breaks = np.random.randint(2, 7)
                distance = np.random.uniform(5.0, 20.0)

                _, y = model.multi_wire_break(
                    distance=distance,
                    num_breaks=breaks,
                    gap=gap,
                    fault_type=fault_type,
                    want_plot=False,
                    data_dir=data_dir,
                )

                X.append(self._pad_or_crop(y))
                y_class.append(class_label)
                y_gap.append(gap)
                y_break_count.append(breaks)
                y_distance.append(distance)

        save_path = os.path.join(save_dir, "multi_dataset.npz")
        np.savez(
            save_path,
            X=np.array(X),
            y_class=np.array(y_class),
            y_gap=np.array(y_gap),
            y_break_count=np.array(y_break_count),
            y_distance=np.array(y_distance),
        )
        print(f"Multi dataset saved to: {save_path}")
