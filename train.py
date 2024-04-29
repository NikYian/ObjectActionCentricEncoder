#!/usr/bin/env python3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import json

from args import Args
from datasets.image_dataset import generate_image_dataset
from datasets.utils import ssv2_id2class, extract_subset
from models.AcE import get_AcE
from models.trainer import AcE_Trainer
from models.utils import get_criterion
from models.teacher import load_teacher


import torch.nn.functional as F

if __name__ == "__main__":

    args = Args()
    dataset, train_loader, val_loader, test_loader = generate_image_dataset(
        args, gen_obj_crops=True, gen_VAE_features=False
    )
    video_cls_dict = ssv2_id2class(args)
    subset_dataset = extract_subset(dataset, [2], [143], video_cls_dict)

    AcE = get_AcE(args).to(args.device)

    with open(args.ssv2_labels, "r") as f:
        ssv2_labels = json.load(f)
        ssv2_labels = {value: key for key, value in ssv2_labels.items()}

    criterion = get_criterion(args)
    optimizer = torch.optim.Adam(AcE.parameters(), lr=args.AcE_lr)

    trainer = AcE_Trainer(
        args,
        AcE,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.device,
        args.log_dir,
    )

    if args.AcE_checkpoint:
        epochs_done = os.path.basename(args.AcE_checkpoint).split("_")[2].split(".")[0]
        epochs_remaining = args.AcE_epochs - int(epochs_done)
        if epochs_remaining > 0:
            trainer.train(num_epochs=args.AcE_epochs)
    else:
        trainer.train(num_epochs=args.AcE_epochs)

    print(trainer.evaluate(test_loader))

    teacher = load_teacher(args)

    for batch in train_loader:
        images = batch[0].to(args.device)
        breakpoint()
