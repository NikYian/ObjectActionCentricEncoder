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


class TeacherTrainer:
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
        epoch_loss = 0
        val_loss = 0
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            for i, (clips, labels, _) in enumerate(self.train_loader):
                clips = clips.to(self.device)
                labels = labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(clips)

                batch_size = outputs.shape[0]
                # loss = self.get_loss(outputs, labels, batch_size)
                targets = self.get_targets(labels, batch_size)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * clips.size(0)

                if i % 1 == 0:
                    self.writer.add_scalar(
                        "training_loss", loss.item(), epoch * len(self.train_loader) + i
                    )
                pbar.set_description(
                    desc=f"Epoch {epoch+1}/{num_epochs},Batch {i}/{len(self.train_loader)} Train Loss: {epoch_loss:.4f},Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f} "
                )

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.writer.add_scalar("epoch_training_loss", epoch_loss, epoch)
            # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

            val_loss = self.evaluate(self.val_loader)
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
                    os.path.join(self.log_dir, "teacher_head_" + str(epoch) + ".pth"),
                )
                # print("Best model saved!")
            pbar.set_description(
                desc=f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f},Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f} "
            )

    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(data_loader)

        with torch.no_grad():
            for clips, labels, _ in data_loader:
                clips = clips.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(clips)
                batch_size = outputs.shape[0]
                targets = self.get_targets(labels, batch_size)
                loss = self.criterion(outputs, targets)
                # loss = self.get_loss(outputs, labels, batch_size)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        self.model.train()

        return avg_loss

    def get_targets(self, labels, batch_size):
        target = torch.full((batch_size, 14), 0.5)
        for batch_i, indx in enumerate(labels):
            if indx < 7:
                target[batch_i, 2 * indx] = 1
                target[batch_i, 2 * indx + 1] = 0
            else:
                target[batch_i, 13] = 1
                target[batch_i, 12] = 0
        return target.to(self.device)

    # def get_loss(self, outputs, labels, batch_size):
    #     rel_output = torch.randn(
    #         batch_size, 2, requires_grad=False
    #     )  # tensor for relevant outputs
    #     targets = torch.randn(batch_size, 2, requires_grad=False)
    #     for batch_i, indx in enumerate(labels):
    #         if indx < 7:
    #             rel_output[batch_i] = outputs[batch_i, 2 * indx : 2 * indx + 2]
    #             targets[batch_i, 0] = 1
    #             targets[batch_i, 1] = 0
    #         else:
    #             rel_output[batch_i] = outputs[batch_i, 12:]
    #             targets[batch_i, 0] = 0
    #             targets[batch_i, 1] = 1
    #     loss = self.criterion(rel_output, targets)
    #     return loss

    def get_acc(self, outputs, labels, batch_size):
        rel_output = torch.randn(
            batch_size, 2, requires_grad=False
        )  # tensor for relevant outputs
        targets = torch.randn(batch_size, 2, requires_grad=False)
        for batch_i, indx in enumerate(labels):
            if indx < 7:
                rel_output[batch_i] = outputs[batch_i, 2 * indx : 2 * indx + 2]
                targets[batch_i, 0] = 1
                targets[batch_i, 1] = 0
            else:
                rel_output[batch_i] = outputs[batch_i, 12:]
                targets[batch_i, 0] = 0
                targets[batch_i, 1] = 1
        loss = self.criterion(rel_output, targets)
        return loss
