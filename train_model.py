import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import torch.nn as nn
import torch.optim as optim
from src.ml.dataset import get_single_dataloaders, get_multi_dataloaders
from src.ml.model import SingleMFLNet, MultiMFLNet


class MFLTrainer:
    """
    Trainer class for Gaussian Probabilistic Single and Multi MFL neural networks.
    """

    def __init__(
        self,
        single_data_path: str,
        multi_data_path: str,
        batch_size: int = 128,
        epochs: int = 50,
        lr: float = 0.001,
    ):
        """
        Initializes the trainer with data loaders and Gaussian NLL loss function.
        """
        self.epochs = epochs

        self.single_train, self.single_test = get_single_dataloaders(
            single_data_path, batch_size=batch_size
        )
        self.multi_train, self.multi_test = get_multi_dataloaders(
            multi_data_path, batch_size=batch_size
        )

        self.single_model = SingleMFLNet()
        self.multi_model = MultiMFLNet()

        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_reg = nn.GaussianNLLLoss(eps=1e-6)

        self.opt_single = optim.Adam(self.single_model.parameters(), lr=lr)
        self.opt_multi = optim.Adam(self.multi_model.parameters(), lr=lr)

    def train_single_model(self) -> None:
        """
        Executes the training loop for the SingleMFLNet using Gaussian NLL.
        """
        print("--- Training Probabilistic SingleMFLNet ---")
        self.single_model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for inputs, labels_class, labels_gap in self.single_train:
                self.opt_single.zero_grad()
                out_class, gap_mu, gap_var = self.single_model(inputs)

                loss_class = self.criterion_class(out_class, labels_class)
                loss_gap = self.criterion_reg(gap_mu, labels_gap, gap_var)
                loss = loss_class + loss_gap

                loss.backward()
                self.opt_single.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:02d}/{self.epochs} | Single Model Loss: {total_loss/len(self.single_train):.4f}"
                )

    def train_multi_model(self) -> None:
        """
        Executes the training loop for the MultiMFLNet using Gaussian NLL.
        """
        print("\n--- Training Probabilistic MultiMFLNet ---")
        self.multi_model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            for (
                inputs,
                labels_class,
                labels_gap,
                labels_count,
                labels_dist,
            ) in self.multi_train:
                self.opt_multi.zero_grad()
                out_class, gap_mu, gap_var, count_mu, count_var, dist_mu, dist_var = (
                    self.multi_model(inputs)
                )

                loss_class = self.criterion_class(out_class, labels_class)
                loss_gap = self.criterion_reg(gap_mu, labels_gap, gap_var)
                loss_count = self.criterion_reg(count_mu, labels_count, count_var)
                loss_dist = self.criterion_reg(dist_mu, labels_dist, dist_var)

                loss = loss_class + loss_gap + loss_count + loss_dist

                loss.backward()
                self.opt_multi.step()
                total_loss += loss.item()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(
                    f"Epoch {epoch+1:02d}/{self.epochs} | Multi Model Loss: {total_loss/len(self.multi_train):.4f}"
                )

    def save_models(self, single_save_path: str, multi_save_path: str) -> None:
        """
        Saves the trained weights of both probabilistic models.
        """
        torch.save(self.single_model.state_dict(), single_save_path)
        torch.save(self.multi_model.state_dict(), multi_save_path)
        print("\nProbabilistic Models successfully trained and saved!")
