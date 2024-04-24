"""
This implementation is based on
https://github.com/MCG-NJU/VideoMAE/blob/main/ssv2.py
pulished under CC-BY-NC 4.0 license: https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE
"""

import os
import numpy as np
import torch
from torchvision import transforms

# from random_erasing import RandomErasing
import warnings
from decord import VideoReader, cpu
from torch.utils.data import Dataset
import externals.VideoMAE.video_transforms as video_transforms
import externals.VideoMAE.volume_transforms as volume_transforms

labels_to_keep = [
    2,  # "Bending something so that it deforms"
    3,  # "Bending something until it breaks"
    5,  # "Closing something"
    14,  # "Folding something"
    22,  # "Letting something roll along a flat surface" --
    134,  # "Something falling like a feather or paper"
    135,  # "Something falling like a rock"
    143,  # "Squeezing something"
    149,  # "Tearing something into two pieces"
    150,  # "Tearing something just a little bit"
    172,  # Unfolding something" --
]

# labels_to_keep = np.arange(0, 174)


class SSVideoClsDataset(Dataset):
    """Load your own video classification dataset."""

    def __init__(
        self,
        anno_path,
        data_path,
        clip_len=8,
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        num_segment=1,
        num_crop=1,
        test_num_segment=10,
        test_num_crop=3,
        args=None,
    ):
        self.anno_path = anno_path
        self.data_path = data_path
        self.clip_len = clip_len
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.num_segment = num_segment
        self.test_num_segment = test_num_segment
        self.num_crop = num_crop
        self.test_num_crop = test_num_crop
        self.args = args
        self.aug = False
        self.rand_erase = False
        if VideoReader is None:
            raise ImportError(
                "Unable to import `decord` which is required to read videos."
            )
        self.get_whole_video = False

        import pandas as pd

        cleaned = pd.read_csv(self.anno_path, header=None, delimiter=" ")
        cleaned_samples = np.array(cleaned.values[:, 0])
        cleaned_labels = np.array(cleaned.values[:, 1])

        indices_to_keep = [
            i for i, label in enumerate(cleaned_labels) if label in labels_to_keep
        ]

        self.dataset_samples = cleaned_samples[indices_to_keep]
        self.label_array = cleaned_labels[indices_to_keep]

        # self.dataset_samples = list(cleaned.values[:, 0])
        # self.label_array = list(cleaned.values[:, 1])

        self.data_transform = video_transforms.Compose(
            [
                video_transforms.Resize(self.short_side_size, interpolation="bilinear"),
                video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                volume_transforms.ClipToTensor(),
                video_transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def get_whole_video_switch(self):
        self.get_whole_video = not self.get_whole_video
        print(f"get_whole_video = {self.get_whole_video}")

    def loadvideo_decord(self, sample, sample_rate_scale=1):
        """Load video content using Decord"""
        fname = sample

        if not (os.path.exists(fname)):
            return []

        # avoid hanging issue
        if os.path.getsize(fname) < 1 * 1024:
            print("SKIP: ", fname, " - ", os.path.getsize(fname))
            return []
        try:
            if self.keep_aspect_ratio:
                vr = VideoReader(fname, num_threads=1, ctx=cpu(0))
            else:
                vr = VideoReader(
                    fname,
                    width=self.new_width,
                    height=self.new_height,
                    num_threads=1,
                    ctx=cpu(0),
                )
        except:
            print("video cannot be loaded by decord: ", fname)
            return []
        # handle temporal segments
        if self.get_whole_video:
            vr.seek(0)
            return vr.get_batch(np.arange(0, len(vr))).asnumpy()
        else:
            average_duration = len(vr) // self.num_segment
            all_index = []
            if average_duration > 0:
                all_index += list(
                    np.multiply(list(range(self.num_segment)), average_duration)
                    + np.random.randint(average_duration, size=self.num_segment)
                )
            elif len(vr) > self.num_segment:
                all_index += list(
                    np.sort(np.random.randint(len(vr), size=self.num_segment))
                )
            else:
                all_index += list(np.zeros((self.num_segment,)))
            all_index = list(np.array(all_index))
            vr.seek(0)
            buffer = vr.get_batch(all_index).asnumpy()
            return buffer

    def __getitem__(self, index):
        sample = self.dataset_samples[index]
        buffer = self.loadvideo_decord(sample)
        if len(buffer) == 0:
            while len(buffer) == 0:
                warnings.warn(
                    "video {} not correctly loaded during validation".format(sample)
                )
                index = np.random.randint(self.__len__())
                sample = self.dataset_samples[index]
                buffer = self.loadvideo_decord(sample)
        if self.get_whole_video:
            buffer = [buffer[i] for i in range(buffer.shape[0])]
        else:
            buffer = self.data_transform(buffer)
        return (buffer, self.label_array[index], sample.split("/")[-1].split(".")[0])

    def __len__(self):
        return len(self.dataset_samples)
