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
from models.trainers.teacher_trainer import TeacherTrainer
from utils.utils import get_criterion
from models.teacher import load_teacher
from datasets.video_dataset import build_video_dataset


import torch.nn.functional as F

if __name__ == "__main__":

    args = Args()

    video_dataset, _ = build_video_dataset(args)
    train_loader, val_loader, test_loader = dataset_split(
        video_dataset, [], args.split_ratios, args.batch_size, video_split=False
    )

    teacher = load_teacher(args)

    # criterion = torch.nn.BCELoss()
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(teacher.parameters(), lr=args.teacher_lr)

    trainer = TeacherTrainer(
        args,
        teacher,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        args.device,
        args.log_dir,
    )

    trainer.train(num_epochs=args.teacher_epochs)

    # for batch in train_loader:

    #     clips = batch[0].to(args.device)
    #     outputs = teacher(clips)

    #     res = teacher.aff(clips)

    #     breakpoint()

    # trainer = AcE_Trainer(
    #     args,
    #     AcE,
    #     train_loader,
    #     val_loader,
    #     criterion,
    #     optimizer,
    #     args.device,
    #     args.log_dir,
    # )

    # if args.AcE_checkpoint:
    #     epochs_done = os.path.basename(args.AcE_checkpoint).split("_")[2].split(".")[0]
    #     epochs_remaining = args.AcE_epochs - int(epochs_done)
    #     if epochs_remaining > 0:
    #         trainer.train(num_epochs=args.AcE_epochs)
    # else:
    #     trainer.train(num_epochs=args.AcE_epochs)

    # trainer.train(num_epochs=args.AcE_epochs)

    # print(f"Test loss: {trainer.evaluate(test_loader)}")

    # # teacher = load_teacher(args)
    # res_top5 = []
    # target_top5 = []
    # for images, target_features, _, _ in test_loader:
    #     images = images.to(args.device)
    #     target_features = target_features.to(args.device)
    #     res = AcE.predict_affordances(images)
    #     target = AcE.ac_head(target_features).topk(k=10, dim=-1).indices
    #     breakpoint()
