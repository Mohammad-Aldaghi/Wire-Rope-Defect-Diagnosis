import torch
import numpy as np
from collections import defaultdict
from src.ml.model import SingleMFLNet, MultiMFLNet


class MFLFusionEngine:
    """
    Advanced Uncertainty-Aware Fusion Engine using Soft-Probability Monte Carlo Dropout
    and Native Gaussian Negative Log-Likelihood variance estimations.
    """

    def __init__(self, single_model_path: str, multi_model_path: str):
        """
        Initializes the fusion engine with probabilistic model weights and fault mappings.
        """
        self.single_model = SingleMFLNet()
        self.single_model.load_state_dict(torch.load(single_model_path, map_location="cpu", weights_only=True))

        self.multi_model = MultiMFLNet()
        self.multi_model.load_state_dict(torch.load(multi_model_path, map_location="cpu", weights_only=True))

        self.fault_mapping = {0: "King", 1: "Inner", 2: "Outer"}

    def _enable_dropout(self, model: torch.nn.Module) -> None:
        """
        Enables dropout layers for Monte Carlo sampling to extract epistemic uncertainty.
        """
        model.eval()
        for module in model.modules():
            if module.__class__.__name__.startswith("Dropout"):
                module.train()

    def predict_scenario(
        self, signal: torch.Tensor, num_samples: int = 100, temperature: float = 3.0
    ) -> list[dict]:
        """
        Executes MC Dropout and aggregates soft probabilities alongside native Gaussian regression parameters.
        Computes the weighted expected value of gap, count, and distance specific to each class.
        """
        self._enable_dropout(self.single_model)
        self._enable_dropout(self.multi_model)

        p_s_all, g_s_all = [], []
        p_m_all, g_m_all, cnt_m_all, d_m_all = [], [], [], []

        with torch.no_grad():
            for _ in range(num_samples):
                out_class_s, gap_mu_s, _ = self.single_model(signal)
                p_s = (
                    torch.nn.functional.softmax(out_class_s / temperature, dim=1)
                    .cpu()
                    .numpy()[0]
                )
                p_s_all.append(p_s)
                g_s_all.append(np.clip(gap_mu_s.item() * 100.0, 1.0, 100.0))

                out_class_m, gap_mu_m, _, count_mu_m, _, dist_mu_m, _ = (
                    self.multi_model(signal)
                )
                p_m = (
                    torch.nn.functional.softmax(out_class_m / temperature, dim=1)
                    .cpu()
                    .numpy()[0]
                )
                p_m_all.append(p_m)
                g_m_all.append(np.clip(gap_mu_m.item() * 100.0, 1.0, 100.0))
                cnt_m_all.append(count_mu_m.item() * 6.0)
                d_m_all.append(np.clip(dist_mu_m.item() * 20.0, 5.0, 20.0))

        p_s_arr = np.array(p_s_all)
        g_s_arr = np.array(g_s_all)
        p_m_arr = np.array(p_m_all)
        g_m_arr = np.array(g_m_all)
        cnt_m_arr = np.array(cnt_m_all)
        d_m_arr = np.array(d_m_all)

        cv_single = np.std(g_s_arr) / (np.mean(g_s_arr) + 1e-6)
        cv_multi = np.std(g_m_arr) / (np.mean(g_m_arr) + 1e-6)

        weight_s = 1.0 / ((cv_single**2) + 1e-4)
        weight_m = 1.0 / ((cv_multi**2) + 1e-4)

        mean_count = np.mean(cnt_m_arr)
        if mean_count < 1.8 or mean_count > 6.2:
            weight_s *= 10.0
            weight_m *= 0.1
        elif 2.0 <= mean_count <= 6.0:
            weight_m *= 2.0

        total_weight = weight_s + weight_m
        w_s_norm = weight_s / total_weight
        w_m_norm = weight_m / total_weight

        prob_s_avg = np.mean(p_s_arr, axis=0)
        prob_m_avg = np.mean(p_m_arr, axis=0)

        results = []

        for c in range(3):
            p_single = w_s_norm * prob_s_avg[c] * 100.0
            if p_single > 1.0:
                weights_c_s = p_s_arr[:, c]
                sum_w_c_s = np.sum(weights_c_s)
                avg_g_s = (
                    np.sum(weights_c_s * g_s_arr) / (sum_w_c_s + 1e-6)
                    if sum_w_c_s > 0
                    else np.mean(g_s_arr)
                )

                results.append(
                    {
                        "scenario": f"Single Break - {self.fault_mapping[c]}",
                        "probability": p_single,
                        "gap_mean": avg_g_s,
                        "dist_mean": 0.0,
                    }
                )

            p_multi = w_m_norm * prob_m_avg[c] * 100.0
            if p_multi > 1.0:
                weights_c_m = p_m_arr[:, c]
                sum_w_c_m = np.sum(weights_c_m)

                avg_g_m = (
                    np.sum(weights_c_m * g_m_arr) / (sum_w_c_m + 1e-6)
                    if sum_w_c_m > 0
                    else np.mean(g_m_arr)
                )
                avg_d_m = (
                    np.sum(weights_c_m * d_m_arr) / (sum_w_c_m + 1e-6)
                    if sum_w_c_m > 0
                    else np.mean(d_m_arr)
                )
                avg_c_m = (
                    int(
                        np.clip(
                            round(np.sum(weights_c_m * cnt_m_arr) / (sum_w_c_m + 1e-6)),
                            2,
                            6,
                        )
                    )
                    if sum_w_c_m > 0
                    else int(np.clip(round(mean_count), 2, 6))
                )

                results.append(
                    {
                        "scenario": f"Multi Break ({avg_c_m}x) - {self.fault_mapping[c]}",
                        "probability": p_multi,
                        "gap_mean": avg_g_m,
                        "dist_mean": avg_d_m,
                    }
                )

        total_prob = sum(r["probability"] for r in results)
        if total_prob > 0:
            for r in results:
                r["probability"] = (r["probability"] / total_prob) * 100.0

        results.sort(key=lambda x: x["probability"], reverse=True)
        return results
