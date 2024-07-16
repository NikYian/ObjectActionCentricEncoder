#!/usr/bin/env python3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
from torch.distributions import MultivariateNormal
from tqdm import tqdm
import numpy as np
from sklearn.metrics import (
    hamming_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from args import Args
from datasets.image_dataset import generate_image_dataset
from models.AcE import AcEnn


def fit_multivariate_gaussians_to_subsets(subsets):
    gaussians_by_affordance = []
    for i in range(10):
        subset = subsets[i]

        if len(subset) == 0:
            gaussians_by_affordance.append(None)
            continue

        subset_array = np.stack(subset)  # Convert list of arrays to a single 2D array
        mean_vector = np.mean(subset_array, axis=0)
        covariance_matrix = np.cov(subset_array, rowvar=False)

        # Ensure covariance matrix is positive semi-definite
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6

        # Define the multivariate normal distribution
        gaussian = MultivariateNormal(
            torch.tensor(mean_vector), torch.tensor(covariance_matrix)
        )
        gaussians_by_affordance.append(gaussian)
    return gaussians_by_affordance


def eval_ACM(args, AcE, ACM, test_loader):
    for (
        clip_features,
        target_features,
        affordance_label,
        multi_label_aff,
        sample_type,
        sample_path,
        object,
    ) in tqdm(test_loader, desc="Evaluating gaussian Affordance Categorization Module"):
        breakpoint()
        clip_features = clip_features.to(args.device)
        predicted_features = AcE.forward_CLIP(clip_features).cpu().detach()
        predictions = []
        for i in range(10):
            gaussian = ACM[i]
            predictions_ = gaussian(predicted_features)
            breakpoint()


def evaluate_multilabel_model(self, y_true, y_pred, y_prob):
    h_loss = hamming_loss(y_true, y_pred)
    subset_acc = accuracy_score(y_true, y_pred)  # Subset accuracy
    precision_micro = precision_score(y_true, y_pred, average="micro")
    recall_micro = recall_score(y_true, y_pred, average="micro", zero_division=np.nan)
    f1_micro = f1_score(y_true, y_pred, average="micro")

    precision_macro = precision_score(
        y_true, y_pred, average="macro", zero_division=np.nan
    )
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=np.nan)
    f1_macro = f1_score(y_true, y_pred, average="macro")

    # print(f"Hamming Loss: {h_loss:.4f}")
    print(f"Subset Accuracy: {subset_acc:.4f}")
    print(f"Micro Precision: {precision_micro:.4f}")
    print(f"Micro Recall: {recall_micro:.4f}")
    print(f"Micro F1 Score: {f1_micro:.4f}")


if __name__ == "__main__":

    args = Args()

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        generate_image_dataset(args)
    )

    AcE = AcEnn(args, head=args.head, checkpoint=args.AcE_checkpoint).to(args.device)
    AcE.eval()

    subsets_by_affordance_gt = [[] for _ in range(10)]
    subsets_by_affordance_ACE = [[] for _ in range(10)]
    for (
        clip_features,
        target_features,
        affordance_label,
        multi_label_aff,
        sample_type,
        sample_path,
        object,
    ) in tqdm(val_dataset, desc="Collecting subsets from training dataset"):
        clip_features = torch.tensor(clip_features).unsqueeze(0).to(args.device)
        predicted_features = AcE.forward_CLIP(clip_features).squeeze(0).cpu().detach()
        predicted_features = predicted_features.numpy()
        subsets_by_affordance_gt[affordance_label].append(target_features)
        subsets_by_affordance_ACE[affordance_label].append(predicted_features)

    subsets_by_affordance_gt_np = [
        np.array(subset) for subset in subsets_by_affordance_gt
    ]
    subsets_by_affordance_ACE_np = [
        np.array(subset) for subset in subsets_by_affordance_ACE
    ]

    # np.save(
    #     "somethings_affordances/subsets_by_affordance_gt.npy",
    #     subsets_by_affordance_gt_np,
    # )
    # np.save(
    #     "somethings_affordances/subsets_by_affordance_ACE.npy",
    #     subsets_by_affordance_ACE_np,
    # )

    print("Fitting gaussians to subsets")
    ACM_gt = fit_multivariate_gaussians_to_subsets(subsets_by_affordance_gt)
    ACM_ACE = fit_multivariate_gaussians_to_subsets(subsets_by_affordance_ACE)
    eval_ACM(args, AcE, ACM_ACE, test_loader)
