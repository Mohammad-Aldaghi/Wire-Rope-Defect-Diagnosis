import torch
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append("../")
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.calibration import calibration_curve
from src.ml.dataset import get_single_dataloaders, get_multi_dataloaders
from src.ml.model import SingleMFLNet, MultiMFLNet
from src.ml.fusion_engine import MFLFusionEngine
from src.physics_model import PhysicsInformedMFL
from src.ml.dataset_generator import DatasetGenerator


class MFLEvaluator:
    def __init__(
        self,
        single_data_path: str,
        multi_data_path: str,
        single_model_path: str,
        multi_model_path: str,
    ):
        _, self.single_test = get_single_dataloaders(single_data_path, batch_size=128)
        _, self.multi_test = get_multi_dataloaders(multi_data_path, batch_size=128)

        self.single_model = SingleMFLNet()
        self.single_model.load_state_dict(torch.load(single_model_path))
        self.single_model.eval()

        self.multi_model = MultiMFLNet()
        self.multi_model.load_state_dict(torch.load(multi_model_path))
        self.multi_model.eval()

        self.fault_map = {0: "King", 1: "Inner", 2: "Outer"}
        self.gen_helper = DatasetGenerator()

    def evaluate_models(self, temperature: float = 3.0) -> None:
        self.y_true_single_class, self.y_pred_single_class = [], []
        self.y_prob_single = []
        self.single_real_gaps, self.single_pred_gaps = [], []
        with torch.no_grad():
            for inputs, labels_class, labels_gap in self.single_test:
                out_class, gap_mu, _ = self.single_model(inputs)
                probs = torch.nn.functional.softmax(out_class / temperature, dim=1)
                self.y_prob_single.extend(probs.cpu().numpy())
                self.y_true_single_class.extend(labels_class.cpu().numpy())
                self.y_pred_single_class.extend(
                    torch.argmax(probs, dim=1).cpu().numpy()
                )
                self.single_real_gaps.extend((labels_gap * 100.0).cpu().numpy())
                self.single_pred_gaps.extend((gap_mu * 100.0).cpu().numpy())

        self.y_true_multi_class, self.y_pred_multi_class = [], []
        self.y_prob_multi = []
        self.multi_real_gaps, self.multi_pred_gaps = [], []
        with torch.no_grad():
            for inputs, labels_class, labels_gap, _, _ in self.multi_test:
                out_class, gap_mu, _, _, _, _, _ = self.multi_model(inputs)
                probs = torch.nn.functional.softmax(out_class / temperature, dim=1)
                self.y_prob_multi.extend(probs.cpu().numpy())
                self.y_true_multi_class.extend(labels_class.cpu().numpy())
                self.y_pred_multi_class.extend(torch.argmax(probs, dim=1).cpu().numpy())
                self.multi_real_gaps.extend((labels_gap * 100.0).cpu().numpy())
                self.multi_pred_gaps.extend((gap_mu * 100.0).cpu().numpy())

    def _apply_gap_tolerance(self, real_gaps: list, pred_gaps: list) -> pd.DataFrame:
        categories, statuses = [], []
        for r, p in zip(real_gaps, pred_gaps):
            err = abs(r - p)
            if r <= 10.0:
                cat = "Small (0-10)"
                status = "Accurate" if err <= 5.0 else "Error"
            elif r <= 60.0:
                cat = "Medium (10-60)"
                status = "Accurate" if err <= 10.0 else "Error"
            else:
                cat = "Large (60-100)"
                status = "Accurate" if err <= 20.0 else "Error"
            categories.append(cat)
            statuses.append(status)
        return pd.DataFrame({"Category": categories, "Status": statuses})

    def plot_gap_tolerance(self) -> None:
        single_df = self._apply_gap_tolerance(
            self.single_real_gaps, self.single_pred_gaps
        )
        multi_df = self._apply_gap_tolerance(self.multi_real_gaps, self.multi_pred_gaps)
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        single_cm = (
            single_df.groupby(["Category", "Status"]).size().unstack(fill_value=0)
        )
        row_order = ["Small (0-10)", "Medium (10-60)", "Large (60-100)"]
        single_cm = single_cm.reindex(row_order).fillna(0)
        sns.heatmap(
            single_cm,
            annot=True,
            fmt="g",
            cmap="Greens",
            cbar=False,
            ax=axes[0],
            annot_kws={"fontsize": 12},
        )
        axes[0].set_title("Single Model: Gap Tolerance Matrix", fontweight="bold")
        axes[0].set_ylabel("True Gap Size")
        axes[0].set_xlabel("Prediction Status")

        multi_cm = multi_df.groupby(["Category", "Status"]).size().unstack(fill_value=0)
        multi_cm = multi_cm.reindex(row_order).fillna(0)
        sns.heatmap(
            multi_cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            cbar=False,
            ax=axes[1],
            annot_kws={"fontsize": 12},
        )
        axes[1].set_title("Multi Model: Gap Tolerance Matrix", fontweight="bold")
        axes[1].set_ylabel("")
        axes[1].set_xlabel("Prediction Status")
        plt.tight_layout()
        plt.show()

    def plot_probabilistic_metrics(self) -> None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        y_true_s_arr = np.array(self.y_true_single_class)
        y_pred_s = np.array(self.y_pred_single_class)
        y_prob_s = np.array(self.y_prob_single)
        y_true_m_arr = np.array(self.y_true_multi_class)
        y_pred_m = np.array(self.y_pred_multi_class)
        y_prob_m = np.array(self.y_prob_multi)

        y_true_s_bin = label_binarize(y_true_s_arr, classes=[0, 1, 2])
        y_true_m_bin = label_binarize(y_true_m_arr, classes=[0, 1, 2])

        prob_true_s, prob_pred_s = calibration_curve(
            y_true_s_bin.ravel(), y_prob_s.ravel(), n_bins=10
        )
        axes[0, 0].plot(
            prob_pred_s, prob_true_s, marker="o", color="purple", label="Single Model"
        )
        axes[0, 0].plot([0, 1], [0, 1], linestyle="--", color="black")
        axes[0, 0].set_title("Single: Calibration Curve", fontweight="bold")

        prob_true_m, prob_pred_m = calibration_curve(
            y_true_m_bin.ravel(), y_prob_m.ravel(), n_bins=10
        )
        axes[1, 0].plot(
            prob_pred_m, prob_true_m, marker="o", color="teal", label="Multi Model"
        )
        axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="black")
        axes[1, 0].set_title("Multi: Calibration Curve", fontweight="bold")

        fpr_s, tpr_s, _ = roc_curve(y_true_s_bin.ravel(), y_prob_s.ravel())
        axes[0, 1].plot(
            fpr_s, tpr_s, color="purple", label=f"AUC = {auc(fpr_s, tpr_s):.3f}"
        )
        axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="black")
        axes[0, 1].set_title("Single: ROC Curve", fontweight="bold")
        axes[0, 1].legend(loc="lower right")

        fpr_m, tpr_m, _ = roc_curve(y_true_m_bin.ravel(), y_prob_m.ravel())
        axes[1, 1].plot(
            fpr_m, tpr_m, color="teal", label=f"AUC = {auc(fpr_m, tpr_m):.3f}"
        )
        axes[1, 1].plot([0, 1], [0, 1], linestyle="--", color="black")
        axes[1, 1].set_title("Multi: ROC Curve", fontweight="bold")
        axes[1, 1].legend(loc="lower right")

        conf_s = np.max(y_prob_s, axis=1)
        correct_s = y_pred_s == y_true_s_arr
        if len(conf_s[correct_s]) > 0:
            sns.histplot(
                conf_s[correct_s],
                bins=20,
                color="green",
                alpha=0.5,
                label="Correct",
                ax=axes[0, 2],
                kde=True,
            )
        if len(conf_s[~correct_s]) > 0:
            sns.histplot(
                conf_s[~correct_s],
                bins=20,
                color="red",
                alpha=0.5,
                label="Incorrect",
                ax=axes[0, 2],
                kde=True,
            )
        axes[0, 2].set_title("Single: Confidence Histogram", fontweight="bold")
        axes[0, 2].legend()

        conf_m = np.max(y_prob_m, axis=1)
        correct_m = y_pred_m == y_true_m_arr
        if len(conf_m[correct_m]) > 0:
            sns.histplot(
                conf_m[correct_m],
                bins=20,
                color="green",
                alpha=0.5,
                label="Correct",
                ax=axes[1, 2],
                kde=True,
            )
        if len(conf_m[~correct_m]) > 0:
            sns.histplot(
                conf_m[~correct_m],
                bins=20,
                color="red",
                alpha=0.5,
                label="Incorrect",
                ax=axes[1, 2],
                kde=True,
            )
        axes[1, 2].set_title("Multi: Confidence Histogram", fontweight="bold")
        axes[1, 2].legend()

        plt.tight_layout()
        plt.show()

    def plot_base_top3(
        self, single_samples: list, multi_samples: list, temperature: float = 3.0
    ) -> None:
        fig, axes = plt.subplots(5, 2, figsize=(20, 25))
        colors = ["red", "orange", "blue"]
        styles = ["--", "-.", ":"]
        for i, (sig, l_class, l_gap) in enumerate(single_samples):
            real_sig = sig.squeeze().numpy()
            real_c = self.fault_map[l_class.item()]
            real_g = l_gap.item() * 100.0
            with torch.no_grad():
                out_class, gap_mu, _ = self.single_model(sig.view(1, 1, -1))
                probs = (
                    torch.nn.functional.softmax(out_class / temperature, dim=1)
                    .squeeze()
                    .numpy()
                )
                pred_g = np.clip(gap_mu.item() * 100.0, 1.0, 100.0)
            ranks = np.argsort(probs)[::-1]
            axes[i, 0].plot(
                real_sig,
                color="black",
                linewidth=3.5,
                label=f"Real (Single - {real_c} | Gap: {real_g:.1f})",
                alpha=0.7,
            )
            for rank_idx, c_idx in enumerate(ranks[:3]):
                fault_str = self.fault_map[c_idx].lower()
                phys_model = PhysicsInformedMFL(
                    f"../data/Inner King Outer dataset/{fault_str}_fault_dataset.npz"
                )
                _, y_raw = phys_model.generate_signal(
                    target_gap=pred_g, sample_size=1000
                )
                pred_sig = self.gen_helper._pad_or_crop(y_raw)
                if np.max(np.abs(pred_sig)) > 0:
                    shift = np.argmax(np.abs(real_sig)) - np.argmax(np.abs(pred_sig))
                    pred_sig = np.roll(pred_sig, shift)
                label = f"Rank {rank_idx+1} ({probs[c_idx]*100:.1f}%): Single - {self.fault_map[c_idx]} | Gap: {pred_g:.1f}"
                axes[i, 0].plot(
                    pred_sig,
                    color=colors[rank_idx],
                    linestyle=styles[rank_idx],
                    linewidth=2.5,
                    label=label,
                )
            axes[i, 0].set_title(f"Base Single Model - Test {i+1}", fontweight="bold")
            axes[i, 0].legend(loc="upper right")

        for i, (sig, l_class, l_gap, l_count, l_dist) in enumerate(multi_samples):
            real_sig = sig.squeeze().numpy()
            real_c = self.fault_map[l_class.item()]
            real_g = l_gap.item() * 100.0
            real_cnt = round(l_count.item() * 6.0)
            real_d = l_dist.item() * 20.0
            with torch.no_grad():
                out_class, gap_mu, _, count_mu, _, dist_mu, _ = self.multi_model(
                    sig.view(1, 1, -1)
                )
                probs = (
                    torch.nn.functional.softmax(out_class / temperature, dim=1)
                    .squeeze()
                    .numpy()
                )
                pred_g = np.clip(gap_mu.item() * 100.0, 1.0, 100.0)
                pred_cnt = int(np.clip(round(count_mu.item() * 6.0), 2, 6))
                pred_d = np.clip(dist_mu.item() * 20.0, 5.0, 20.0)
            ranks = np.argsort(probs)[::-1]
            axes[i, 1].plot(
                real_sig,
                color="black",
                linewidth=3.5,
                label=f"Real (Multi {real_cnt}x - {real_c} | Gap: {real_g:.1f} | Dist: {real_d:.1f})",
                alpha=0.7,
            )
            for rank_idx, c_idx in enumerate(ranks[:3]):
                fault_str = self.fault_map[c_idx].lower()
                phys_model = PhysicsInformedMFL(
                    f"../data/Inner King Outer dataset/{fault_str}_fault_dataset.npz"
                )
                _, y_raw = phys_model.multi_wire_break(
                    distance=pred_d,
                    num_breaks=pred_cnt,
                    gap=pred_g,
                    fault_type=fault_str,
                    want_plot=False,
                    data_dir="../data/Inner King Outer dataset",
                )
                pred_sig = self.gen_helper._pad_or_crop(y_raw)
                if np.max(np.abs(pred_sig)) > 0:
                    shift = np.argmax(np.abs(real_sig)) - np.argmax(np.abs(pred_sig))
                    pred_sig = np.roll(pred_sig, shift)
                label = f"Rank {rank_idx+1} ({probs[c_idx]*100:.1f}%): Multi {pred_cnt}x - {self.fault_map[c_idx]} | Gap: {pred_g:.1f}"
                axes[i, 1].plot(
                    pred_sig,
                    color=colors[rank_idx],
                    linestyle=styles[rank_idx],
                    linewidth=2.5,
                    label=label,
                )
            axes[i, 1].set_title(f"Base Multi Model - Test {i+1}", fontweight="bold")
            axes[i, 1].legend(loc="upper right")
        plt.tight_layout()
        plt.show()


class ComprehensiveFusionEvaluator:
    def __init__(self, single_model_path: str, multi_model_path: str):
        self.fusion = MFLFusionEngine(single_model_path, multi_model_path)
        self.fault_map = {0: "King", 1: "Inner", 2: "Outer"}
        self.macro_classes = [
            "Single - King",
            "Single - Inner",
            "Single - Outer",
            "Multi - King",
            "Multi - Inner",
            "Multi - Outer",
        ]
        self.gen_helper = DatasetGenerator()

    def _get_macro_probabilities(self, scenarios: list[dict]) -> list[float]:
        probs = {macro: 0.0 for macro in self.macro_classes}
        for sc in scenarios:
            name = sc["scenario"]
            domain = "Single" if "Single Break" in name else "Multi"
            fault = name.split(" - ")[-1]
            macro_key = f"{domain} - {fault}"
            if macro_key in probs:
                probs[macro_key] += sc["probability"] / 100.0
        return [probs[macro] for macro in self.macro_classes]

    def run_statistical_evaluation(
        self, single_test, multi_test, num_samples: int = 100
    ) -> None:
        single_samples = random.sample(list(single_test), num_samples)
        multi_samples = random.sample(list(multi_test), num_samples)
        self.y_true, self.y_prob = [], []
        self.categories, self.statuses = [], []
        self.correct_confidences, self.incorrect_confidences = [], []
        for sig, l_class, l_gap in single_samples:
            real_c = f"Single - {self.fault_map[l_class.item()]}"
            real_g = l_gap.item() * 100.0
            scenarios = self.fusion.predict_scenario(sig.view(1, 1, -1), num_samples=30)
            self.y_true.append(self.macro_classes.index(real_c))
            self.y_prob.append(self._get_macro_probabilities(scenarios))
            self._apply_tolerance(real_g, scenarios[0]["gap_mean"])
            self._track_confidence(real_c, scenarios[0])
        for sig, l_class, l_gap, l_count, l_dist in multi_samples:
            real_c = f"Multi - {self.fault_map[l_class.item()]}"
            real_g = l_gap.item() * 100.0
            scenarios = self.fusion.predict_scenario(sig.view(1, 1, -1), num_samples=30)
            self.y_true.append(self.macro_classes.index(real_c))
            self.y_prob.append(self._get_macro_probabilities(scenarios))
            self._apply_tolerance(real_g, scenarios[0]["gap_mean"])
            self._track_confidence(real_c, scenarios[0])

    def _apply_tolerance(self, real_g: float, pred_g: float) -> None:
        err = abs(real_g - pred_g)
        if real_g <= 10.0:
            cat, status = "Small (0-10)", "Accurate" if err <= 5.0 else "Error"
        elif real_g <= 60.0:
            cat, status = "Medium (10-60)", "Accurate" if err <= 7.0 else "Error"
        else:
            cat, status = "Large (60-100)", "Accurate" if err <= 20.0 else "Error"
        self.categories.append(cat)
        self.statuses.append(status)

    def _track_confidence(self, real_c: str, top_scenario: dict) -> None:
        pred_c = (
            "Single - " + top_scenario["scenario"].split(" - ")[-1]
            if "Single" in top_scenario["scenario"]
            else "Multi - " + top_scenario["scenario"].split(" - ")[-1]
        )
        conf = top_scenario["probability"] / 100.0
        if real_c == pred_c:
            self.correct_confidences.append(conf)
        else:
            self.incorrect_confidences.append(conf)

    def plot_comprehensive_metrics(self) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        df = pd.DataFrame({"Category": self.categories, "Status": self.statuses})
        cm = df.groupby(["Category", "Status"]).size().unstack(fill_value=0)
        row_order = ["Small (0-10)", "Medium (10-60)", "Large (60-100)"]
        cm = cm.reindex(row_order).fillna(0)
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="YlGnBu",
            cbar=False,
            ax=axes[0, 0],
            annot_kws={"fontsize": 12},
        )
        axes[0, 0].set_title("Fusion Engine: Gap Tolerance Matrix", fontweight="bold")
        axes[0, 0].set_ylabel("True Gap Size")
        axes[0, 0].set_xlabel("Prediction Status")

        y_true_bin = label_binarize(self.y_true, classes=[0, 1, 2, 3, 4, 5])
        y_prob_arr = np.array(self.y_prob)

        prob_true, prob_pred = calibration_curve(
            y_true_bin.ravel(), y_prob_arr.ravel(), n_bins=10
        )
        axes[0, 1].plot(
            prob_pred, prob_true, marker="o", color="purple", label="Fusion Model"
        )
        axes[0, 1].plot([0, 1], [0, 1], linestyle="--", color="black")
        axes[0, 1].set_title("Fusion: Calibration Curve", fontweight="bold")

        fpr, tpr, _ = roc_curve(y_true_bin.ravel(), y_prob_arr.ravel())
        axes[1, 0].plot(
            fpr, tpr, color="teal", label=f"Micro-Average AUC = {auc(fpr, tpr):.3f}"
        )
        axes[1, 0].plot([0, 1], [0, 1], linestyle="--", color="black")
        axes[1, 0].set_title("Fusion: ROC Curve", fontweight="bold")
        axes[1, 0].legend(loc="lower right")

        if len(self.correct_confidences) > 0:
            sns.histplot(
                self.correct_confidences,
                bins=15,
                color="green",
                alpha=0.5,
                label="Correct",
                ax=axes[1, 1],
                kde=True,
            )
        if len(self.incorrect_confidences) > 0:
            sns.histplot(
                self.incorrect_confidences,
                bins=15,
                color="red",
                alpha=0.5,
                label="Incorrect",
                ax=axes[1, 1],
                kde=True,
            )
        axes[1, 1].set_title("Fusion: Confidence Distribution", fontweight="bold")
        axes[1, 1].legend()
        plt.tight_layout()
        plt.show()

    def plot_top_k_bar_chart(
        self, single_test, multi_test, num_samples: int = 100
    ) -> None:
        """
        Generates a bar chart displaying the Top-1, Top-2, and Top-3 accuracy
        using the 6 macro classes (Single/Multi - Inner/King/Outer).
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        import random

        single_samples = random.sample(list(single_test), num_samples)
        multi_samples = random.sample(list(multi_test), num_samples)

        top1_acc, top2_acc, top3_acc = 0, 0, 0
        total = num_samples * 2

        def process_samples(samples: list, domain_prefix: str) -> None:
            """
            Evaluates the samples against the 6 macro-class predictions.
            """
            nonlocal top1_acc, top2_acc, top3_acc
            for sig, l_class, *_ in samples:
                real_label = f"{domain_prefix} - {self.fault_map[l_class.item()]}"
                scenarios = self.fusion.predict_scenario(
                    sig.view(1, 1, -1), num_samples=30
                )
                scenarios_labels = []
                for sc in scenarios:
                    domain = "Single" if "Single" in sc["scenario"] else "Multi"
                    cls = sc["scenario"].split(" - ")[-1]
                    scenarios_labels.append(f"{domain} - {cls}")

                if len(scenarios_labels) > 0 and scenarios_labels[0] == real_label:
                    top1_acc += 1

                if len(scenarios_labels) > 1 and real_label in scenarios_labels[:2]:
                    top2_acc += 1
                elif len(scenarios_labels) > 0 and real_label == scenarios_labels[0]:
                    top2_acc += 1

                if len(scenarios_labels) > 2 and real_label in scenarios_labels[:3]:
                    top3_acc += 1
                elif len(scenarios_labels) > 0 and real_label in scenarios_labels[:2]:
                    top3_acc += 1

        process_samples(single_samples, "Single")
        process_samples(multi_samples, "Multi")

        top1_pct = (top1_acc / total) * 100.0
        top2_pct = (top2_acc / total) * 100.0
        top3_pct = (top3_acc / total) * 100.0

        plt.figure(figsize=(8, 6))
        ax = sns.barplot(
            x=["Top-1", "Top-2", "Top-3"],
            y=[top1_pct, top2_pct, top3_pct],
            palette="magma",
        )
        plt.title("Top-K Predictive Accuracy", fontweight="bold", fontsize=14)
        plt.ylabel("Accuracy (%)", fontweight="bold")
        plt.ylim(0, 110)

        for i, v in enumerate([top1_pct, top2_pct, top3_pct]):
            ax.text(i, v + 2, f"{v:.1f}%", ha="center", fontweight="bold", fontsize=12)

        plt.tight_layout()
        plt.show()

    def plot_fusion_top3(self, single_samples: list, multi_samples: list) -> None:
        # Changed to 1 row, 2 columns for a single random test
        fig, axes = plt.subplots(1, 2, figsize=(20, 5))
        colors = ["red", "orange", "blue"]

        # Format the samples
        all_samples = [
            (sig, l_class, l_gap, 1, 0.0) for sig, l_class, l_gap in single_samples
        ]
        all_samples_multi = [
            (sig, l_class, l_gap, l_count, l_dist)
            for sig, l_class, l_gap, l_count, l_dist in multi_samples
        ]

        # Pick just ONE random sample for each column
        random_single = [random.choice(all_samples)]
        random_multi = [random.choice(all_samples_multi)]

        for col, samples in enumerate([random_single, random_multi]):
            for i, (sig, l_class, l_gap, l_count, l_dist) in enumerate(samples):
                real_sig = sig.squeeze().numpy()
                real_c = self.fault_map[l_class.item()]
                real_g = l_gap.item() * 100.0

                if isinstance(l_count, int) and col == 0:
                    real_label = f"Real (Single - {real_c} | Gap: {real_g:.1f})"
                else:
                    real_cnt = round(l_count.item() * 6.0)
                    real_d = l_dist.item() * 20.0
                    real_label = f"Real (Multi {real_cnt}x - {real_c} | Gap: {real_g:.1f} | Dist: {real_d:.1f})"

                scenarios = self.fusion.predict_scenario(
                    sig.view(1, 1, -1), num_samples=30
                )

                axes[col].plot(
                    real_sig, color="black", linewidth=3.5, label=real_label, alpha=0.7
                )

                for rank_idx in range(min(3, len(scenarios))):
                    sc = scenarios[rank_idx]
                    pred_g = np.clip(sc["gap_mean"], 1.0, 100.0)
                    fault_str = sc["scenario"].split(" - ")[-1].lower()

                    phys_model = PhysicsInformedMFL(
                        f"../data/Inner King Outer dataset/{fault_str}_fault_dataset.npz"
                    )

                    if "Single Break" in sc["scenario"]:
                        _, y_raw = phys_model.generate_signal(
                            target_gap=pred_g, sample_size=1000
                        )
                        # Label for Single Break
                        label = f"Rank {rank_idx+1} ({sc['probability']:.1f}%): {sc['scenario']} | Gap: {pred_g:.1f}"
                    else:
                        pred_b = int(sc["scenario"].split("(")[1].split("x")[0])
                        pred_d = np.clip(sc["dist_mean"], 5.0, 20.0)
                        _, y_raw = phys_model.multi_wire_break(
                            distance=pred_d,
                            num_breaks=pred_b,
                            gap=pred_g,
                            fault_type=fault_str,
                            want_plot=False,
                            data_dir="../data/Inner King Outer dataset",
                        )
                        # Label for Multi Break (Now includes Distance!)
                        label = f"Rank {rank_idx+1} ({sc['probability']:.1f}%): {sc['scenario']} | Gap: {pred_g:.1f} | Dist: {pred_d:.1f}"

                    pred_sig = self.gen_helper._pad_or_crop(y_raw)
                    if np.max(np.abs(pred_sig)) > 0:
                        shift = np.argmax(np.abs(real_sig)) - np.argmax(
                            np.abs(pred_sig)
                        )
                        pred_sig = np.roll(pred_sig, shift)

                    axes[col].plot(
                        pred_sig,
                        color=colors[rank_idx],
                        linewidth=1.5,
                        label=label,
                    )

                domain = "Single" if col == 0 else "Multi"
                axes[col].set_title(
                    f"Fusion Engine on True {domain}", fontweight="bold"
                )
                axes[col].legend(loc="upper right")

        plt.tight_layout()
        plt.show()

    def plot_macro_confusion_matrix(self) -> None:
        """
        Generates and displays a 4x4 confusion matrix for the Inner and Outer macro classes,
        completely omitting any King categories.
        """
        # Redefine macro classes locally to strictly exclude King
        pure_macro_classes = [
            "Single - Inner",
            "Single - Outer",
            "Multi - Inner",
            "Multi - Outer",
        ]

        y_pred = np.argmax(self.y_prob, axis=1)

        keep = [1, 2, 4, 5]   # Single-Inner, Single-Outer, Multi-Inner, Multi-Outer
        y_true = np.array(self.y_true); y_pred = np.argmax(self.y_prob, axis=1)
        mask = np.isin(y_true, keep)
        cm = confusion_matrix(y_true[mask], y_pred[mask], labels=keep)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Purples",
            xticklabels=pure_macro_classes,
            yticklabels=pure_macro_classes,
            annot_kws={"fontsize": 12, "fontweight": "bold"},
        )
        plt.title(
            "Fusion Engine: 4x4 Macro-Class Confusion Matrix",
            fontweight="bold",
            fontsize=13,
        )
        plt.ylabel("True Fault", fontweight="bold")
        plt.xlabel("Predicted Fault", fontweight="bold")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()
