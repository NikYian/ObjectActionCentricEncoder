#!/usr/bin/env python3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch

from args import Args
from datasets.image_dataset import generate_image_dataset
from models.AcE import AcEnn
from models.trainers.AcE_trainer import AcE_Trainer
from utils.utils import get_criterion


import torch.nn.functional as F

if __name__ == "__main__":

    args = Args()

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        generate_image_dataset(args)
    )

    AcE = AcEnn(args, head=args.head).to(args.device)
    criterion = get_criterion(args)
    optimizer = torch.optim.Adam(
        AcE.parameters(), lr=args.AcE_lr, weight_decay=args.AcE_weight_decay
    )
    name = "AcE_" + args.head
    trainer = AcE_Trainer(
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
