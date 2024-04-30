import pandas as pd
import numpy as np
import os
from torch.utils.data import Subset
import torch
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

from datasets.image_dataset import OAcEImgDataset


class ssv2_id2class:
    def __init__(self, args):
        self.dict = {}
        cleaned = pd.read_csv(args.anno_path, header=None, delimiter=" ")
        paths = np.array(cleaned.values[:, 0])
        labels = np.array(cleaned.values[:, 1])

        for i, label in enumerate(labels):
            video_id = os.path.basename(paths[i]).split(".")[0]
            self.dict[video_id] = label

    def cls(self, video_id):
        return self.dict[video_id]


def extract_subset(dataset, object_ids, video_cls_ls, video_cls_dict):
    indices = []
    video_ids = set()
    for index, sample in enumerate(dataset):

        video_cls = video_cls_dict.cls(sample[3])
        if sample[2] in object_ids and video_cls in video_cls_ls:
            video_ids.add(sample[3])
            indices.append(index)

    subset_dataset = Subset(dataset, indices)
    return subset_dataset, video_ids


def dataset_split(dataset, video_ids, ratios, batch_size, video_split=True):
    torch.manual_seed(42)
    indices = torch.randperm(len(dataset))

    if video_split:
        video_ids = np.array(list(video_ids))
        video_indices = torch.randperm(len(video_ids))
        train_ratio = ratios[0]
        val_ratio = ratios[1]

        train_size = int(train_ratio * len(video_ids))
        val_size = int(val_ratio * len(video_ids))

        train_vid_indices = video_indices[:train_size]
        val_vid_indices = video_indices[train_size : train_size + val_size]
        test_vid_indices = video_indices[train_size + val_size :]
        train_video_ids = video_ids[train_vid_indices]
        val_video_ids = video_ids[val_vid_indices]
        test_video_ids = video_ids[test_vid_indices]

        train_indices = []
        val_indices = []
        test_indices = []
        for i, sample in enumerate(dataset):
            if sample[3] in train_video_ids:
                train_indices.append(i)
            elif sample[3] in val_video_ids:
                val_indices.append(i)
            elif sample[3] in test_video_ids:
                test_indices.append(i)

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
        val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=batch_size)
        test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size)

    else:
        train_ratio = ratios[0]
        val_ratio = ratios[1]

        train_size = int(train_ratio * len(dataset))
        val_size = int(val_ratio * len(dataset))

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        test_sampler = SubsetRandomSampler(test_indices)

        train_loader = DataLoader(dataset, sampler=train_sampler, batch_size=batch_size)
        val_loader = DataLoader(dataset, sampler=val_sampler, batch_size=batch_size)
        test_loader = DataLoader(dataset, sampler=test_sampler, batch_size=batch_size)
    return train_loader, val_loader, test_loader
