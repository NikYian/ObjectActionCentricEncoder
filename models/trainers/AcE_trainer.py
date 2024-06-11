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
    roc_auc_score,
)


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
        # best_val_loss = float("inf")
        best_val_acc = 0
        epoch_loss = 0
        val_acc = 0
        scheduler = lr_scheduler.StepLR(
            self.optimizer, step_size=10, gamma=0.5, verbose=True
        )

        pbar = tqdm(
            range(num_epochs),
        )
        for epoch in pbar:
            self.model.train()
            running_loss = 0.0

            for i, (
                clip_features,
                target_features,
                affordance_labels,
                multi_label_targets,
                sample_type,
                sample_path,
                _,
            ) in enumerate(self.train_loader):
                self.model.update_aff_anchors()
                self.optimizer.zero_grad()

                clip_features = clip_features.to(self.device)
                target_features = target_features.to(self.device)
                outputs, similarities = self.model.forward_CLIP_action_pred(
                    clip_features, affordance_labels
                )
                loss_i = self.criterion(outputs, target_features)

                multi_label_targets = (
                    torch.stack(multi_label_targets).transpose(0, 1).to(self.device)
                )
                # MSE = nn.MSELoss()
                loss_t = self.criterion(similarities, multi_label_targets)

                # action_pred = similarities[
                #     torch.arange(len(clip_features)), affordance_labels
                # ]
                # mask = [sample == "i" for sample in sample_type]
                # filtered_action_pred = action_pred[mask]
                # targets = torch.ones_like(filtered_action_pred)
                # # MSE = nn.MSELoss()
                # loss_t = self.criterion(filtered_action_pred, targets)
                print(f"losst {loss_t}")
                print(f"lossi {loss_i}")

                loss = 0.3 * loss_t + loss_i
                # loss = loss_i
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() * clip_features.size(0)

                if i % 1 == 0:
                    self.writer.add_scalar(
                        "training_loss",
                        loss.item(),
                        epoch * len(self.train_loader) + i,
                    )

                pbar.set_description(
                    desc=f"Batch {i}/{len(self.train_loader)}. Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f},Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f} "
                )

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.writer.add_scalar("epoch_training_loss", epoch_loss, epoch)
            val_acc, _ = self.evaluate(self.val_loader)
            print(similarities[:3])
            # breakpoint()
            # print(self.model.temperatures)
            self.writer.add_scalar("val_acc", val_acc, epoch)
            scheduler.step()
            pbar.set_description(
                desc=f"Batch {i}/{len(self.train_loader)}. Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f},Val Acc: {val_acc:.4f}, Best Val Acc: {best_val_acc:.4f} "
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc

            pth_files = glob.glob(os.path.join(self.log_dir, "*.pth"))
            for pth_file in pth_files:
                os.remove(pth_file)

            torch.save(
                self.model.head.state_dict(),
                os.path.join(self.log_dir, "AcE_head_" + str(epoch) + ".pth"),
            )

    def evaluate(self, data_loader, threshold=0.7, brk=False):
        self.model.eval()
        self.model.update_aff_anchors()
        # total_loss = 0.0
        # num_batches = len(data_loader)

        # total_characteristics_sum = np.zeros(8)
        total_samples = 0

        pred_list = []
        binary_pred_list = []
        target_list = []

        with torch.no_grad():
            for (
                clip_features,
                _,
                _,
                multi_label_targets,
                _,
                sample_path,
                object,
            ) in data_loader:
                clip_features = clip_features.to(self.device)
                multi_label_targets = torch.stack(multi_label_targets, dim=1).float()
                multi_label_targets = multi_label_targets.to(self.device)

                batch_size = clip_features.size(0)
                total_samples += batch_size

                AcE_features = self.model.forward_CLIP(clip_features)

                predictions = self.model.ZS_predict(AcE_features, cosine=True)
                if brk:
                    breakpoint()
                # threshold = 0.9
                binary_predictions = (predictions > threshold).int()

                pred_list.append(predictions.cpu().numpy())
                binary_pred_list.append(binary_predictions.cpu().numpy())
                target_list.append(multi_label_targets.cpu().numpy())

        pred_list = np.concatenate(pred_list, axis=0)
        binary_pred_list = np.concatenate(binary_pred_list, axis=0)
        target_list = np.concatenate(target_list, axis=0)

        accuracy = np.mean(binary_pred_list == target_list)
        print(f"Accuracy: {accuracy:.4f}")

        # self.evaluate_multilabel_model(target_list, binary_pred_list, pred_list)

        self.model.train()

        return accuracy, target_list
        # all_targets,

    def evaluate_multilabel_model(self, y_true, y_pred, y_prob):
        h_loss = hamming_loss(y_true, y_pred)
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

        print(f"Hamming Loss: {h_loss:.4f}")
        print(f"Subset Accuracy: {subset_acc:.4f}")
        print(f"Micro Precision: {precision_micro:.4f}")
        print(f"Micro Recall: {recall_micro:.4f}")
        print(f"Micro F1 Score: {f1_micro:.4f}")
        # print(f"Micro ROC-AUC: {roc_auc_micro:.4f}")
        print(f"Macro Precision: {precision_macro:.4f}")
        print(f"Macro Recall: {recall_macro:.4f}")
        print(f"Macro F1 Score: {f1_macro:.4f}")
        # print(f"Macro ROC-AUC: {roc_auc_macro:.4f}")

    def print_aff(self, predictions):
        for affordance, prediction in zip(self.args.affordances, predictions):
            print(f"{affordance}: {prediction:.2f}")
