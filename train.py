#!/usr/bin/env python3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import torch
import torch.nn as nn

from args import Args
from datasets.image_dataset import generate_image_dataset
from models.AcE import AcE
from models.trainer import AcE_Trainer

import torch.nn.functional as F

if __name__ == "__main__":

    args = Args()
    dataset, train_loader, val_loader, test_loader = generate_image_dataset(
        args, gen_obj_crops=False, gen_VAE_features=False
    )
    AcE = AcE(args).to(args.device)

    criterion = F.mse_loss
    optimizer = torch.optim.Adam(AcE.parameters(), lr=0.0001)

    trainer = AcE_Trainer(
        AcE, train_loader, val_loader, criterion, optimizer, args.device, args.log_dir
    )

    trainer.train(num_epochs=args.AcE_epochs)

    print(trainer.evaluate(test_loader))
