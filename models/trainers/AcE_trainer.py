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
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from models.utils import get_scheduler


class AcE_Trainer:
    def __init__(
        self,
        args,
        name,
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="logs",
    ):
        self.name = name
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.args = args
        self.scheduler = get_scheduler(self.args, self.optimizer, self.train_loader)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self, num_epochs):
        val_loss = self.evaluate(self.val_loader)
        best_val_loss = val_loss
        epoch_loss = 0

        pbar = tqdm(
            range(num_epochs),
        )
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            total_samples = 0

            for i, (
                clip_features,
                target_features,
                affordance_labels,
                multi_label_targets,
                sample_type,
                sample_path,
                _,
            ) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                clip_features = clip_features.to(self.device)
                target_features = target_features.to(self.device)

                AcE_features = self.model.forward_image_features(clip_features)
                loss = self.criterion(AcE_features, target_features)
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                total_samples += clip_features.size(0)

                running_loss += loss.item() * clip_features.size(0)
                epoch_loss = running_loss / total_samples
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

                if i % 1 == 0:
                    self.writer.add_scalar(
                        "training_loss",
                        loss.item(),
                        epoch * len(self.train_loader) + i,
                    )

                pbar.set_description(
                    desc=f"lr = {lr:.4f} Epoch {epoch+1}/{num_epochs} | Batch {i}/{len(self.train_loader)} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_loss:.4f} | Best Val Acc: {best_val_loss:.4f}"
                )

            self.writer.add_scalar("epoch_training_loss", epoch_loss, epoch)
            val_loss = self.evaluate(self.val_loader)

            self.writer.add_scalar("val_loss", val_loss, epoch)

            if val_loss < best_val_loss:
                best_val_loss = val_loss

            # pth_files = glob.glob(os.path.join(self.log_dir, "*.pth"))
            # for pth_file in pth_files:
            #     os.remove(pth_file)
            path = os.path.join(self.log_dir, self.name + ".pth")
            torch.save(
                self.model.head.state_dict(),
                path,
            )
            print(f"Model saved at {path}")

    def evaluate(
        self,
        data_loader,
    ):
        self.model.eval()
        total_samples = 0
        running_loss = 0

        with torch.no_grad():
            for (
                clip_features,
                target_features,
                _,
                _,
                _,
                _,
                _,
            ) in data_loader:
                clip_features = clip_features.to(self.device)
                target_features = target_features.to(self.device)
                batch_size = clip_features.size(0)
                total_samples += batch_size

                AcE_features = self.model.forward_image_features(clip_features)
                loss = self.criterion(AcE_features, target_features)
                running_loss += loss.item() * batch_size

        self.model.train()

        return running_loss / total_samples if total_samples > 0 else 0
