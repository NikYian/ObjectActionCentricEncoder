#!/usr/bin/env python3
import os

from args import Args
from datasets.image_dataset import generate_image_dataset


args = Args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


if __name__ == "__main__":

    args = Args()

    generate_image_dataset(args)

    # TODO: check if dataset is created then load . if not create then load
    # TODO: initalize AcE
    # TODO: train AcE
