# This code has been adapted and modified from  https://github.com/gorkaydemir/SOLV
import os

# import random
import numpy as np
import torch
import json
import random
from torch.utils import data
from glob import glob
from PIL import Image
import sys
from math import ceil, floor

sys.path.append(r"./externals")
from SOLV.datasets.ytvis19 import remove_borders, To_One_Hot

import torchvision.transforms as T
import torch.nn.functional as F


# === Dataset Classes ===
class SSV2(data.Dataset):
    def __init__(self, args, split, sa_labels):
        self.root = args.root
        self.sa_labels = sa_labels

        self.N = args.N
        self.relative_orders = list(range(-self.N, self.N + 1))

        self.resize_to = args.resize_to

        self.patch_size = args.patch_size
        self.token_num = args.token_num

        # === Get Video Names and Lengths ===
        self.dataset_list = []
        self.video_lengths = []
        # self.split_name = []

        json_fname = split + "_videos.json"

        json_path = os.path.join(self.root, json_fname)
        videos = json.load(open(json_path))

        for video in videos:

            self.dataset_list.append(video["id"])
            self.video_lengths.append(video["length"])
            # self.split_name.append("train")

        self.create_idx_frame_mapping()

        # === Transformations ===
        self.resize = T.Resize(self.resize_to)
        self.resize_nn = T.Resize(self.resize_to, T.InterpolationMode.NEAREST)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def transform(self, image):

        # image, _ = remove_borders(image)

        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        return image

    def create_idx_frame_mapping(self):
        self.mapping = []

        for video_idx, video_length in enumerate(self.video_lengths):
            video_id = self.dataset_list[video_idx]
            # split_name = self.split_name[video_idx]
            for video_frame_idx in range(video_length):
                path = self.create_frame_path(video_id, video_frame_idx)
                if path in self.sa_labels[video_id]["good_frames"]:
                    indx = self.sa_labels[video_id]["good_frames"].index(path)
                    bb = self.sa_labels[video_id]["bbs"][indx]
                    self.mapping.append((video_id, video_frame_idx, bb))

    def create_frame_path(self, video_id: str, frame_index: int) -> str:
        frame_str = f"{frame_index:04d}"
        return f"{video_id}/{frame_str}.jpg"

    def get_rgb(self, idx):
        video_id, frame_idx, bb = self.mapping[idx]
        img_dir = os.path.join(self.root, "jpg", video_id)
        img_list = sorted(
            glob(os.path.join(img_dir, "*.jpg")),
            key=lambda x: int(x.split("/")[-1].split(".")[0]),
        )
        frame_num = len(img_list)
        target_frame_path = img_list[2]
        input_frames = torch.zeros(
            (2 * self.N + 1, 3, self.resize_to[0], self.resize_to[1]), dtype=torch.float
        )
        mask = torch.ones(2 * self.N + 1)

        for i, frame_order in enumerate(self.relative_orders):
            frame_idx_real = frame_idx + frame_order

            if frame_idx_real < 0 or frame_idx_real >= frame_num:
                mask[i] = 0
                continue

            frame_raw = Image.open(img_list[frame_idx_real]).convert("RGB")

            frame = self.transform(frame_raw)
            input_frames[i] = frame
        original_height, original_width = self.to_tensor(frame_raw).shape[1:]
        new_height, new_width = self.resize_to
        scale_factor_height = new_height / original_height
        scale_factor_width = new_width / original_width
        # Scale bounding box
        left, upper, right, lower = bb
        scaled_left = floor(left * scale_factor_width)
        scaled_upper = floor(upper * scale_factor_height)
        scaled_right = ceil(right * scale_factor_width)
        scaled_lower = ceil(lower * scale_factor_height)
        bb = (scaled_left, scaled_upper, scaled_right, scaled_lower)
        bb_mask = torch.zeros((new_height, new_width), dtype=torch.bool)
        bb_mask[scaled_upper:scaled_lower, scaled_left:scaled_right] = True

        return input_frames, mask, video_id, frame_idx, bb_mask, target_frame_path

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        """
        :return:
            input_features: RGB frames [t-N, ..., t+N]
                            in shape (2*N + 1, 3, H, W)

            frame_masks: Mask for input_features indicating if frame is available
                            in shape (2*N + 1)
        """

        (input_frames, frame_masks, video_id, frame_idx, bb_mask, target_frame_path) = (
            self.get_rgb(idx)
        )  # (2N + 1, 3, H, W), (2N + 1)

        affordance_label = self.sa_labels[video_id]["affordance"]
        multi_label_aff = self.sa_labels[video_id]["affordance_labels"]
        multi_label_aff[affordance_label] = 4
        # sa_labels = self.sa_labels[video_id]
        # bb = torch.tensor(bb)
        multi_label_aff = torch.tensor(multi_label_aff)
        return (
            input_frames,
            frame_masks,
            video_id,
            frame_idx,
            multi_label_aff,
            bb_mask,
            target_frame_path,
        )


class SSV2_features(data.Dataset):
    def __init__(self, args, split, sa_labels):
        self.root = args.root
        self.sa_labels = sa_labels

        # === Get Video Names and Lengths ===
        self.dataset_list = []

        json_fname = split + "_videos.json"

        json_path = os.path.join(self.root, json_fname)
        videos = json.load(open(json_path))

        for video in videos:

            self.dataset_list.append(video["id"])

    def transform(self, image):

        # image, _ = remove_borders(image)

        image = self.resize(image)
        image = self.to_tensor(image)
        image = self.normalize(image)

        return image

    def create_frame_path(self, video_id: str, frame_index: int) -> str:
        frame_str = f"{frame_index:04d}"
        return f"{video_id}/{frame_str}.jpg"

    def get_rgb(self, idx):
        video_id, frame_idx, bb = self.mapping[idx]
        img_dir = os.path.join(self.root, "jpg", video_id)
        img_list = sorted(
            glob(os.path.join(img_dir, "*.jpg")),
            key=lambda x: int(x.split("/")[-1].split(".")[0]),
        )
        frame_num = len(img_list)
        target_frame_path = img_list[2]
        input_frames = torch.zeros(
            (2 * self.N + 1, 3, self.resize_to[0], self.resize_to[1]), dtype=torch.float
        )
        mask = torch.ones(2 * self.N + 1)

        for i, frame_order in enumerate(self.relative_orders):
            frame_idx_real = frame_idx + frame_order

            if frame_idx_real < 0 or frame_idx_real >= frame_num:
                mask[i] = 0
                continue

            frame_raw = Image.open(img_list[frame_idx_real]).convert("RGB")

            frame = self.transform(frame_raw)
            input_frames[i] = frame
        original_height, original_width = self.to_tensor(frame_raw).shape[1:]
        new_height, new_width = self.resize_to
        scale_factor_height = new_height / original_height
        scale_factor_width = new_width / original_width
        # Scale bounding box
        left, upper, right, lower = bb
        scaled_left = floor(left * scale_factor_width)
        scaled_upper = floor(upper * scale_factor_height)
        scaled_right = ceil(right * scale_factor_width)
        scaled_lower = ceil(lower * scale_factor_height)
        bb = (scaled_left, scaled_upper, scaled_right, scaled_lower)
        bb_mask = torch.zeros((new_height, new_width), dtype=torch.bool)
        bb_mask[scaled_upper:scaled_lower, scaled_left:scaled_right] = True

        return input_frames, mask, video_id, frame_idx, bb_mask, target_frame_path

    def __len__(self):
        return len(self.mapping)

    def __getitem__(self, idx):
        """
        :return:
            input_features: RGB frames [t-N, ..., t+N]
                            in shape (2*N + 1, 3, H, W)

            frame_masks: Mask for input_features indicating if frame is available
                            in shape (2*N + 1)
        """

        (input_frames, frame_masks, video_id, frame_idx, bb_mask, target_frame_path) = (
            self.get_rgb(idx)
        )  # (2N + 1, 3, H, W), (2N + 1)

        affordance_label = self.sa_labels[video_id]["affordance"]
        multi_label_aff = self.sa_labels[video_id]["affordance_labels"]
        multi_label_aff[affordance_label] = 4
        # sa_labels = self.sa_labels[video_id]
        # bb = torch.tensor(bb)
        multi_label_aff = torch.tensor(multi_label_aff)
        return (
            input_frames,
            frame_masks,
            video_id,
            frame_idx,
            multi_label_aff,
            bb_mask,
            target_frame_path,
        )
