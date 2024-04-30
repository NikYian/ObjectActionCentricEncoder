#!/usr/bin/env python3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import json

from args import Args
from datasets.image_dataset import generate_image_dataset
from datasets.utils import ssv2_id2class, extract_subset, dataset_split
from models.AcE import get_AcE
from models.trainer import AcE_Trainer
from models.utils import get_criterion
from models.teacher import load_teacher


import torch.nn.functional as F

if __name__ == "__main__":

    args = Args()
    dataset = generate_image_dataset(args, gen_obj_crops=False, gen_VAE_features=False)

    video_cls_dict = ssv2_id2class(args)

    # subset 1 consists of examples of bottles(obj id =2) being squeezed (action id = 143)
    print("Generating subset 1")
    subset1, s1_video_ids = extract_subset(
        dataset, object_ids=[2], video_cls_ls=[143], video_cls_dict=video_cls_dict
    )
    print("Generating subset 2")
    # subset 2 consists of examples of bottles(obj id =2) being rolled (action id = 143)
    subset2, s2_video_ids = extract_subset(
        dataset, object_ids=[2], video_cls_ls=[122, 143], video_cls_dict=video_cls_dict
    )

    train_loader, val_loader, test_loader = dataset_split(
        subset2, s2_video_ids, args.split_ratios, args.AcE_batch_size
    )

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

    # if args.AcE_checkpoint:
    #     epochs_done = os.path.basename(args.AcE_checkpoint).split("_")[2].split(".")[0]
    #     epochs_remaining = args.AcE_epochs - int(epochs_done)
    #     if epochs_remaining > 0:
    #         trainer.train(num_epochs=args.AcE_epochs)
    # else:
    #     trainer.train(num_epochs=args.AcE_epochs)

    # trainer.train(num_epochs=args.AcE_epochs)

    print(f"Test loss: {trainer.evaluate(test_loader)}")

    # teacher = load_teacher(args)
    res_top5 = []
    target_top5 = []
    for images, target_features, _, _ in test_loader:
        images = images.to(args.device)
        target_features = target_features.to(args.device)
        res = AcE.predict_affordances(images)
        target = AcE.ac_head(target_features).topk(k=10, dim=-1).indices
        breakpoint()