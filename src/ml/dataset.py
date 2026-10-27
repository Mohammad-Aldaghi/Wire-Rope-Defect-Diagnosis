"""
MFL PyTorch Dataset and DataLoader Setup
========================================
This module provides custom PyTorch Dataset classes for Single and Multi MFL signals
and helper functions to create DataLoaders.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class SingleMFLDataset(Dataset):
    def __init__(self, X: np.ndarray, y_class: np.ndarray, y_gap: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_gap = torch.tensor(y_gap, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y_class[idx], self.y_gap[idx]


class MultiMFLDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y_class: np.ndarray,
        y_gap: np.ndarray,
        y_break_count: np.ndarray,
        y_distance: np.ndarray,
    ):
        self.X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
        self.y_class = torch.tensor(y_class, dtype=torch.long)
        self.y_gap = torch.tensor(y_gap, dtype=torch.float32)
        self.y_break_count = torch.tensor(y_break_count, dtype=torch.float32)
        self.y_distance = torch.tensor(y_distance, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.X[idx],
            self.y_class[idx],
            self.y_gap[idx],
            self.y_break_count[idx],
            self.y_distance[idx],
        )


def get_single_dataloaders(
    npz_path: str, batch_size: int = 32, test_size: float = 0.2, random_state: int = 42
) -> tuple[DataLoader, DataLoader]:
    dataset = np.load(npz_path)
    X = dataset["X"]
    y_class = dataset["y_class"]
    y_gap = dataset["y_gap"] / 100.0

    X_train, X_test, yc_train, yc_test, yg_train, yg_test = train_test_split(
        X,
        y_class,
        y_gap,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class,
    )

    train_loader = DataLoader(
        SingleMFLDataset(X_train, yc_train, yg_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        SingleMFLDataset(X_test, yc_test, yg_test), batch_size=batch_size, shuffle=False
    )

    return train_loader, test_loader


def get_multi_dataloaders(
    npz_path: str, batch_size: int = 32, test_size: float = 0.2, random_state: int = 42
) -> tuple[DataLoader, DataLoader]:
    dataset = np.load(npz_path)
    X = dataset["X"]
    y_class = dataset["y_class"]
    y_gap = dataset["y_gap"] / 100.0
    y_break_count = dataset["y_break_count"] / 6.0
    y_distance = dataset["y_distance"] / 20.0

    (
        X_train,
        X_test,
        yc_train,
        yc_test,
        yg_train,
        yg_test,
        yb_train,
        yb_test,
        yd_train,
        yd_test,
    ) = train_test_split(
        X,
        y_class,
        y_gap,
        y_break_count,
        y_distance,
        test_size=test_size,
        random_state=random_state,
        stratify=y_class,
    )

    train_loader = DataLoader(
        MultiMFLDataset(X_train, yc_train, yg_train, yb_train, yd_train),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        MultiMFLDataset(X_test, yc_test, yg_test, yb_test, yd_test),
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, test_loader
