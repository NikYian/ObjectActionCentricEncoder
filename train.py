#!/usr/bin/env python3
import os
import deepspeed

from models.teacher import load_teacher
from args import Args

args = Args()


from config.dataset_config import get_dataset_config


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from datasets.video_dataset import build_video_dataset
import torch
from PIL import Image
import numpy as np


def generate_dataset(args):

    video_dataset, video_dataloader = build_video_dataset(args)
    video_dataset.get_whole_video_switch()
    sample = video_dataset[200]

    pil_img = Image.fromarray(sample[3])
    pil_img.save("image.jpg")

    breakpoint()

    # TODO: for every batch extract first frame (ff) and teacher high level features (tf)
    # TODO: create dataset from the above x=ff y=tf


if __name__ == "__main__":

    args = Args()

    generate_dataset(args)

    # TODO: check if dataset is created then load . if not create then load
    # TODO: initalize AcE
    # TODO: train AcE
