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


class AcE_Trainer:
    def __init__(
        self,
        args,
        model,
        train_loader,
        val_loader,
        criterion,
        val_criterion,
        optimizer,
        device="cuda" if torch.cuda.is_available() else "cpu",
        log_dir="logs",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.val_criterion = val_criterion
        self.optimizer = optimizer
        self.device = device
        self.log_dir = log_dir
        self.args = args

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self, num_epochs):
        best_val_loss = float("inf")
        epoch_loss = 0
        val_loss = 0

        # characteristics_means = np.zeros((num_epochs, 8))

        pbar = tqdm(
            range(num_epochs),
        )
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0
            for i, (clip_features, target_features, _, _) in enumerate(
                self.train_loader
            ):
                clip_features = clip_features.to(self.device)
                target_features = target_features.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model.forward_CLIP(clip_features)
                loss = self.criterion(outputs, target_features)

                # # aff_sentence = self.args.affordance_sentences[aff_label]
                # text_outputs = self.model.forward_text(aff_sentence)
                # text_loss = self.criterion(text_outputs, features)

                # loss = image_loss + text_loss
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * clip_features.size(0)

                if i % 1 == 0:
                    self.writer.add_scalar(
                        "training_loss", loss.item(), epoch * len(self.train_loader) + i
                    )

                pbar.set_description(
                    desc=f"Batch {i}/{len(self.train_loader)}. Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f},Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f} "
                )

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.writer.add_scalar("epoch_training_loss", epoch_loss, epoch)

            # print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}")

            val_loss = self.evaluate(self.val_loader)
            self.writer.add_scalar("val_loss", val_loss, epoch)

            # for i, characteristic in enumerate(
            #     self.args.affordance_teacher_decoder.keys()
            # ):
            #     self.writer.add_scalar(characteristic, characteristics_means[i], epoch)

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
                # pbar.set_description(
                #     desc=f"tr = {characteristics_means[3]:.2f},ts = {characteristics_means[5]:.2f}, Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f},Val Loss: {val_loss:.4f}, Best Val Loss: {best_val_loss:.4f} "
                # )

    def evaluate(self, data_loader):
        self.model.eval()
        self.model.update_aff_anchors()
        total_loss = 0.0
        num_batches = len(data_loader)

        # total_characteristics_sum = np.zeros(8)
        total_samples = 0

        with torch.no_grad():
            for clip_features, _, _, multi_label_targets in data_loader:
                clip_features = clip_features.to(self.device)
                multi_label_targets = torch.stack(multi_label_targets, dim=1).float()
                multi_label_targets = multi_label_targets.to(self.device)

                batch_size = clip_features.size(0)
                total_samples += batch_size

                AcE_features = self.model.forward_CLIP(clip_features)

                predictions = self.model.ZS_predict(AcE_features)

                # total_characteristics_sum += np.sum(res, axis=0)

                # outputs = self.model(images)

                loss = self.val_criterion(predictions, multi_label_targets)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        # characteristics_means = total_characteristics_sum / total_samples
        self.model.train()

        return avg_loss  # , characteristics_means
