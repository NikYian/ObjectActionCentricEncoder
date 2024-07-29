#!/usr/bin/env python3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch

from args import Args
from datasets.image_dataset import generate_image_dataset
from models.AcE import AcEnn
from models.trainers.ACM_trainer import ACM_trainer
from utils.utils import get_criterion


if __name__ == "__main__":

    args = Args()

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        generate_image_dataset(args)
    )

    AcE = AcEnn(
        args,
        head=args.head,
        ACM_features=args.ACM_features,
        image_features=args.image_features,
    ).to(args.device)
    for param in AcE.head.parameters():  # AcE params are frozen
        param.requires_grad = False
    criterion = get_criterion(args)
    optimizer = torch.optim.Adam(
        AcE.parameters(), lr=args.AcE_lr, weight_decay=args.AcE_weight_decay
    )
    name = "ACM_" + args.ACM_type + "_" + args.image_features
    trainer = ACM_trainer(
        args,
        name,
        AcE,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.device,
        args.log_dir,
    )
    trainer.train(num_epochs=args.AcE_epochs)

    # print("Finetuning thresholds")
    # y_true, y_pred_probs = trainer.evaluate(trainer.val_loader, return_lists=True)
    # AcE.thresholds = trainer.fine_tune_thresholds(y_true, y_pred_probs)
    # print(f"Threshold finetunig completed:{AcE.thresholds}")
    trainer.evaluate(test_loader, test=True)

    trainer.extract_examples(test_loader)
