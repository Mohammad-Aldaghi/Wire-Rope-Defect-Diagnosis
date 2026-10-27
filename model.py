import torch
import torch.nn as nn
import torch.nn.functional as F


class SingleMFLNet(nn.Module):
    """
    1D CNN for Single Wire Break signals.
    Outputs classification logits and probabilistic Gaussian parameters (mu, variance) for gap regression.
    """

    def __init__(self):
        """
        Initializes the architecture with probabilistic regression heads.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 15, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.head_class = nn.Linear(256, 3)
        self.head_gap = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning class logits, gap mean (mu), and gap variance.
        """
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        shared_out = self.shared_fc(features_flat)

        out_class = self.head_class(shared_out)
        gap_out = self.head_gap(shared_out)

        gap_mu = gap_out[:, 0]
        gap_var = F.softplus(gap_out[:, 1]) + 1e-6

        return out_class, gap_mu, gap_var


class MultiMFLNet(nn.Module):
    """
    1D CNN for Multi Wire Break signals.
    Outputs classification logits and probabilistic Gaussian parameters for gap, count, and distance.
    """

    def __init__(self):
        """
        Initializes the MultiMFLNet architecture with probabilistic regression heads.
        """
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.shared_fc = nn.Sequential(
            nn.Linear(64 * 15, 256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.head_class = nn.Linear(256, 3)
        self.head_gap = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
        self.head_count = nn.Sequential(nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2))
        self.head_distance = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(), nn.Linear(64, 2)
        )

    def forward(self, x: torch.Tensor) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Forward pass returning logits and (mu, var) pairs for all regression targets.
        """
        features = self.feature_extractor(x)
        features_flat = features.view(features.size(0), -1)
        shared_out = self.shared_fc(features_flat)

        out_class = self.head_class(shared_out)

        gap_out = self.head_gap(shared_out)
        gap_mu = gap_out[:, 0]
        gap_var = F.softplus(gap_out[:, 1]) + 1e-6

        count_out = self.head_count(shared_out)
        count_mu = count_out[:, 0]
        count_var = F.softplus(count_out[:, 1]) + 1e-6

        dist_out = self.head_distance(shared_out)
        dist_mu = dist_out[:, 0]
        dist_var = F.softplus(dist_out[:, 1]) + 1e-6

        return out_class, gap_mu, gap_var, count_mu, count_var, dist_mu, dist_var
