"""
This implementation is based on
https://github.com/MCG-NJU/VideoMAE/blob/main/ssv2.py
pulished under CC-BY-NC 4.0 license: https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE
"""

import os
from datasets.ssv2 import SSVideoClsDataset
import torch


def build_video_dataset(args, video_ids):
    anno_path = args.anno_path
    dataset = SSVideoClsDataset(
        video_ids=video_ids,
        data_path=args.data_path,
        clip_len=1,
        num_segment=args.num_frames,
        test_num_segment=args.test_num_segment,
        test_num_crop=args.test_num_crop,
        num_crop=1,
        keep_aspect_ratio=True,
        crop_size=args.input_size,
        short_side_size=args.short_side_size,
        new_height=256,
        new_width=320,
        args=args,
    )
    nb_classes = 174

    assert nb_classes == args.nb_classes

    # sampler_test = torch.utils.data.SequentialSampler(dataset)
    sampler_test = torch.utils.data.RandomSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler_test,
        batch_size=args.batch_size,
        num_workers=1,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    return dataset, data_loader
