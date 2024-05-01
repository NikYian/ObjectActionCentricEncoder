import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F
import json
from tqdm import tqdm
import glob


class AcE_Trainer:
    def __init__(
        self,
        args,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="logs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.args = args

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self, num_epochs):
        best_val_loss = float("inf")

        pbar = tqdm(
            range(num_epochs),
        )
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            for i, (images, labels, _, _) in enumerate(self.train_loader):
                images = images.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * images.size(0)

                if i % 1 == 0:
                    self.writer.add_scalar(
                        "training_loss", loss.item(), epoch * len(self.train_loader) + i
                    )

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.writer.add_scalar("epoch_training_loss", epoch_loss, epoch)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

            val_loss, ts, tr = self.evaluate(self.val_loader)
            self.writer.add_scalar("val_loss", val_loss, epoch)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")

            # Save the best model based on validation accuracy
            if val_loss < best_val_loss:
                best_val_loss = val_loss

                pth_files = glob.glob(os.path.join(self.log_dir, "*.pth"))
                for pth_file in pth_files:
                    os.remove(pth_file)

                torch.save(
                    self.model.head.state_dict(),
                    os.path.join(self.log_dir, "AcE_head_" + str(epoch) + ".pth"),
                )
                # print("Best model saved!")
            pbar.set_description(
                desc=f"tr = {tr:.2f},ts = {ts:.2f}, Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f},Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f} "
            )

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(data_loader)

        with torch.no_grad():
            for images, target_features, _, _ in data_loader:
                images = images.to(self.device)
                target_features = target_features.to(self.device)

                import numpy as np

                res = self.model.predict_affordances(images)
                total_squeezableness = np.mean(res[:, 5])
                total_rollablenesss = np.mean(res[:, 3])

                outputs = self.model(images)

                loss = self.criterion(outputs, target_features)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.model.train()

        return avg_loss, total_squeezableness, total_rollablenesss
