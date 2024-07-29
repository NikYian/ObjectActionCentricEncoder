import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from sklearn.metrics import accuracy_score
from torch.utils.tensorboard import SummaryWriter
from utils.utils import get_scheduler, add_text_to_image, create_image_grid
import os
import torch.nn.functional as F
import json
from tqdm import tqdm
import glob
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)


class ACM_trainer:
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
        self.model = model.to(device) if model else None
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.num_classes = len(args.affordances)
        self.device = device
        self.log_dir = log_dir
        self.args = args
        self.scheduler = (
            get_scheduler(self.args, self.optimizer, self.train_loader)
            if train_loader
            else None
        )

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self, num_epochs):
        print(f"Training ACM with {self.args.ACM_features} features")
        for param in self.model.head.parameters():
            param.requires_grad = False
        epoch_loss = 0
        print("Baseline evaluation:")
        best_val_acc = val_acc = self.evaluate(self.val_loader)
        print("Training starts:")
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
                multi_label_targets,
                _,
            ) in enumerate(self.train_loader):
                self.optimizer.zero_grad()

                clip_features = clip_features.to(self.device)
                target_features = target_features.to(self.device)
                multi_label_targets = 1 * (
                    torch.stack(multi_label_targets).transpose(0, 1).to(self.device)
                )
                if self.args.ACM_features == "gt":
                    predictions = self.model.predict_aff_image_features(target_features)
                else:
                    predictions = self.model.predict_aff_image_features(clip_features)

                loss = self.criterion(predictions.float(), multi_label_targets.float())

                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                running_loss += loss.item() * clip_features.size(0)
                total_samples += clip_features.size(0)
                epoch_loss = running_loss / total_samples
                lr = self.optimizer.state_dict()["param_groups"][0]["lr"]

                if i % 1 == 0:
                    self.writer.add_scalar(
                        "training_loss",
                        loss.item(),
                        epoch * len(self.train_loader) + i,
                    )

                pbar.set_description(
                    desc=f"lr = {lr:.6f} | Batch {i}/{len(self.train_loader)} | Train Loss: {epoch_loss:.4f} | Val Acc: {val_acc:.4f} | Best Val Acc: {best_val_acc:.4f}"
                )

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.writer.add_scalar("epoch_training_loss", epoch_loss, epoch)
            val_acc = self.evaluate(self.val_loader)
            # print(predictions[:5])
            self.writer.add_scalar("val_acc", val_acc, epoch)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
        path = os.path.join(
            self.log_dir, self.name + "_" + self.args.ACM_features + ".pth"
        )
        torch.save(
            self.model.classification_head.state_dict(),
            path,
        )

        print(f"Training completed. Model saved at {path} Finetuning thresholds")
        y_true, y_pred_probs = self.evaluate(self.val_loader, return_lists=True)
        self.model.thresholds = self.fine_tune_thresholds(
            y_true, y_pred_probs, num_classes=self.num_classes
        )
        print(f"Threshold finetunig completed:{self.model.thresholds}")

    def evaluate(
        self,
        data_loader,
        return_f1=False,
        return_lists=False,
        test=False,
    ):
        self.model.eval()

        total_samples = 0

        y_pred_probs = []
        y_binary_list = []
        y_true = []

        with torch.no_grad():
            for (
                clip_features,
                target_features,
                multi_label_targets,
                _,
            ) in tqdm(data_loader):
                clip_features = clip_features.to(self.device)
                target_features = target_features.to(self.device)
                multi_label_targets = (
                    torch.stack(multi_label_targets).transpose(0, 1).to(self.device)
                )
                multi_label_targets = torch.where(
                    multi_label_targets >= 1,
                    torch.tensor(1, device=self.device),
                    multi_label_targets,
                )
                if self.args.ACM_features == "gt":
                    predictions = self.model.predict_aff_image_features(target_features)
                else:
                    predictions = self.model.predict_aff_image_features(clip_features)

                batch_size = clip_features.size(0)
                total_samples += batch_size

                binary_predictions = (predictions > self.model.thresholds).int()

                y_pred_probs.append(predictions.cpu().numpy())
                y_binary_list.append(binary_predictions.cpu().numpy())
                y_true.append(multi_label_targets.cpu().numpy())

        y_pred_probs = np.concatenate(y_pred_probs, axis=0)
        y_binary_list = np.concatenate(y_binary_list, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        accuracy = np.mean(y_binary_list == y_true)

        if test:
            self.evaluate_multilabel_model(y_true, y_binary_list)

        self.model.train()

        if return_f1:
            return f1_score(y_true, y_binary_list, average="micro")
        elif return_lists:
            return y_true, y_pred_probs
        else:
            return accuracy

    def evaluate_multilabel_model(self, y_true, y_pred):
        accuracy = np.mean(y_pred == y_true)
        subset_acc = accuracy_score(y_true, y_pred)  # Subset accuracy
        precision_micro = precision_score(y_true, y_pred, average="micro")
        recall_micro = recall_score(
            y_true, y_pred, average="micro", zero_division=np.nan
        )
        f1_micro = f1_score(y_true, y_pred, average="micro")

        precision_macro = precision_score(
            y_true, y_pred, average="macro", zero_division=np.nan
        )
        recall_macro = recall_score(
            y_true, y_pred, average="macro", zero_division=np.nan
        )
        f1_macro = f1_score(y_true, y_pred, average="macro")

        # Print the overall metrics
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Subset Accuracy: {subset_acc:.4f}")
        print(f"Micro Precision: {precision_micro:.4f}")
        print(f"Micro Recall: {recall_micro:.4f}")
        print(f"Micro F1 Score: {f1_micro:.4f}")
        print(f"Macro Precision: {precision_macro:.4f}")
        print(f"Macro Recall: {recall_macro:.4f}")
        print(f"Macro F1 Score: {f1_macro:.4f}")

        affordances = self.args.affordances
        for i, affordance in enumerate(affordances):
            precision = precision_score(
                y_true[:, i], y_pred[:, i], zero_division=np.nan
            )
            recall = recall_score(y_true[:, i], y_pred[:, i], zero_division=np.nan)
            f1 = f1_score(y_true[:, i], y_pred[:, i], zero_division=np.nan)
            print(
                f"{affordance} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}"
            )

    def print_aff(self, predictions, prnt=True):
        dict = {}
        for affordance, prediction in zip(self.args.affordances, predictions):
            if prnt:
                print(f"{affordance}: {prediction:.2f}")
            dict[affordance] = prediction
        return dict

    def fine_tune_thresholds(
        self, y_true, y_pred_probs, num_classes=10, step_size=0.01
    ):
        best_thresholds = np.zeros(num_classes)
        best_f1 = -1

        for i in range(num_classes):
            best_f1 = -1
            best_threshold = 0.3
            for threshold in np.arange(0.3, 0.9 + step_size, step_size):
                y_pred_binary = (y_pred_probs[:, i] > threshold).astype(int)
                f1 = f1_score(y_true[:, i], y_pred_binary, average="macro")
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            best_thresholds[i] = best_threshold

        return torch.tensor(best_thresholds).to(self.args.device)

    def extract_examples(self, dataloader):
        with torch.no_grad():
            for (
                clip_features,
                _,
                multi_label_targets,
                metadata,
            ) in dataloader:

                clip_features = clip_features.to(self.device)
                multi_label_targets = (
                    torch.stack(multi_label_targets).transpose(0, 1).to(self.device)
                )
                multi_label_targets = torch.where(
                    multi_label_targets >= 1,
                    torch.tensor(1, device=self.device),
                    multi_label_targets,
                )
                predictions = self.model.predict_aff_image_features(clip_features)
                predictions_dict = [
                    self.print_aff(prediction, prnt=False)
                    for prediction in predictions[:15]
                ]
                image_paths = metadata["image_path"][:15]
                output_path = "output_image_grid.jpg"
                create_image_grid(image_paths, predictions_dict, output_path)
                for i, path in enumerate(metadata["image_path"]):
                    print(f"path: {path}")
                    dict_aff = self.print_aff(predictions[i])
                    add_text_to_image(path, "test.jpg", dict_aff)
                    breakpoint()
