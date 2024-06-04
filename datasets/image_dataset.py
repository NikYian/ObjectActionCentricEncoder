from PIL import Image
import json
import os
import clip
import torch
import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader

# from datasets.video_dataset import build_video_dataset
from models.teacher import load_teacher
import externals.VideoMAE.video_transforms as video_transforms
import externals.VideoMAE.volume_transforms as volume_transforms


class OAcEImgDataset(Dataset):
    def __init__(self, image_ids, args, sa_labels, main_objects):
        self.args = args
        _, self.preprocess = clip.load(args.CLIP_model, device=args.device)

        self.image_ids = image_ids
        self.main_objects = main_objects
        self.sa_labels = sa_labels

    def __getitem__(self, index):
        sample_path = self.image_path(self.image_ids[index])
        video_id = self.image_ids[index].split("/")[0]
        image = Image.open(sample_path)
        image_tensor = self.preprocess(image)
        features_path = self.VAE_feature_path(video_id)
        features = np.load(features_path)
        affordance_label = self.sa_labels[video_id]["affordance"]
        affordance_sentense = self.args.affordance_sentences[affordance_label]
        multi_label_aff = self.sa_labels[video_id]["affordance_labels"]

        return (
            image_tensor,
            features,
            affordance_label,
            multi_label_aff,
            affordance_sentense,
        )

    def __len__(self):
        return len(self.image_ids)

    def image_path(self, image_id):
        return os.path.join(self.args.obj_crop_dir, image_id)

    def VAE_feature_path(self, video_id):
        fname = video_id + ".npy"
        return os.path.join(self.args.VAE_features_dir, fname)


def generate_image_dataset(args):

    print("Importing dataset sample ids")

    with open(args.sa_sample_ids["train"], "r") as f:
        train_ids = json.load(f)
    with open(args.sa_sample_ids["val"], "r") as f:
        val_ids = json.load(f)
    with open(args.sa_sample_ids["test"], "r") as f:
        test_ids = json.load(f)

    with open("ssv2/somethings_affordances/sa_labels.json", "r") as f:
        sa_labels = json.load(f)

    with open("ssv2/somethings_affordances/main_objects.json", "r") as f:
        main_objects = json.load(f)

    train_dataset = OAcEImgDataset(train_ids, args, sa_labels, main_objects)
    val_dataset = OAcEImgDataset(val_ids, args, sa_labels, main_objects)
    test_dataset = OAcEImgDataset(test_ids, args, sa_labels, main_objects)

    train_loader = DataLoader(
        train_dataset, batch_size=args.AcE_batch_size, shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=args.AcE_batch_size, shuffle=False)
    test_loader = DataLoader(
        test_dataset, batch_size=args.AcE_batch_size, shuffle=False
    )

    return (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    )
