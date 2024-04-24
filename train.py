#!/usr/bin/env python3
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from args import Args
from datasets.image_dataset import generate_image_dataset

if __name__ == "__main__":

    args = Args()
    generate_image_dataset(args, gen_obj_crops=True)

    # TODO: check if dataset is created then load . if not create then load
    # TODO: initalize AcE
    # TODO: train AcE
