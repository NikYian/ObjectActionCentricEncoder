import pandas as pd
import numpy as np
import os
from torch.utils.data import Subset

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
    for index, sample in enumerate(dataset):
        video_cls = video_cls_dict.cls(sample[3])
        if sample[2] in object_ids and video_cls in video_cls_ls:
            indices.append(index)

    subset_dataset = Subset(dataset, indices)
    breakpoint()
