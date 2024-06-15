#!/usr/bin/env python3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import json
import numpy as np

from args import Args
from datasets.image_dataset import generate_image_dataset
from datasets.utils import ssv2_id2class, extract_subset, dataset_split
from models.AcE import AcEnn
from models.trainers.AcE_trainer import AcE_Trainer
from models.utils import get_criterion
from models.teacher import load_teacher


import torch.nn.functional as F

if __name__ == "__main__":

    args = Args()

    train_dataset, train_loader, val_dataset, val_loader, test_dataset, test_loader = (
        generate_image_dataset(args)
    )

    AcE = AcEnn(args).to(args.device)

    criterion = get_criterion(args)
    val_criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        AcE.parameters(), lr=args.AcE_lr, weight_decay=args.AcE_weight_decay
    )

    with open("ssv2/somethings_affordances/choose_from_N.json", "r") as json_file:
        choose_from_N = json.load(json_file)

    trainer = AcE_Trainer(
        args,
        AcE,
        train_loader,
        val_loader,
        criterion,
        val_criterion,
        optimizer,
        choose_from_N,
        args.device,
        args.log_dir,
    )

    # trainer.evaluate(test_loader, threshold=0.8, clip=True, eval=True)

    # _, all_targets, pred_list = trainer.evaluate(test_loader)
    # num_zeros = np.sum(all_targets == 0)
    # percentage_zeros = (num_zeros / len(all_targets)) * 10
    # print(f"precentage_zeros  target = {percentage_zeros}")
    # num_zeros = np.sum(pred_list == 0)
    # percentage_zeros = (num_zeros / len(pred_list)) * 10
    # print(f"val precentage_zeros pred = {percentage_zeros}")

    # trainer.train(num_epochs=args.AcE_epochs, supervised=False)
    clip = False
    supervised = False
    trainer.train(num_epochs=args.AcE_epochs, supervised=supervised)
    print(0.6)
    test_loss, _, _ = trainer.evaluate(
        test_loader, threshold=0.6, eval=True, clip=clip, supervised=supervised
    )
    print(0.7)
    test_loss, _, _ = trainer.evaluate(
        test_loader, threshold=0.7, eval=True, clip=clip, supervised=supervised
    )
    print(0.8)
    test_loss, _, _ = trainer.evaluate(
        test_loader, threshold=0.8, eval=True, clip=clip, supervised=supervised
    )
    print(0.9)
    test_loss, _, _ = trainer.evaluate(
        test_loader, threshold=0.9, eval=True, clip=clip, supervised=supervised
    )

    # test_loss, _, _ = trainer.evaluate(test_loader, threshold=0.8, eval=True)

    # trainer.choose_from_N_test()
