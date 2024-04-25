import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
import os
import torch.nn.functional as F


class AcE_Trainer:
    def __init__(
        self,
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

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self, num_epochs):
        best_val_sim = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            running_loss = 0.0
            for i, (images, labels, obj_ids) in enumerate(self.train_loader):
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
            print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

            val_sim = self.evaluate(self.val_loader)
            self.writer.add_scalar("val_similarity", val_sim, epoch)
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Accuracy: {val_sim.item():.4f}")

            # Save the best model based on validation accuracy
            if val_sim > best_val_sim:
                best_val_sim = val_sim
                torch.save(
                    self.model.head.state_dict(),
                    os.path.join(self.log_dir, "AcE_head_checkpoint.pth"),
                )
                print("Best model saved!")

    def evaluate(self, data_loader):
        self.model.eval()
        with torch.no_grad():
            for images, target_features, labels in data_loader:
                images = images.to(self.device)
                target_features = labels.to(self.device)

                outputs = self.model(images)

        similarity = F.mse_loss(outputs, target_features)

        return similarity
