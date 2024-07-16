from PIL import Image
import json
import os
import clip
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# from datasets.video_dataset import build_video_dataset
from models.teacher import load_teacher
import externals.VideoMAE.video_transforms as video_transforms
import externals.VideoMAE.volume_transforms as volume_transforms


class SubsetRandomSampler(torch.utils.data.Sampler):
    """Samples elements randomly from a given list of indices, without replacement."""

    def __init__(self, data_source, subset_ratio=0.2):
        self.data_source = data_source
        self.num_samples = int(len(data_source) * subset_ratio)

    def __iter__(self):
        return iter(torch.randperm(len(self.data_source)).tolist()[: self.num_samples])

    def __len__(self):
        return self.num_samples


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


class ImageFeaturesDataset(Dataset):
    def __init__(self, feature_ids, args, sa_labels, main_objects):
        self.args = args

        self.feature_ids = feature_ids
        self.sample_paths = [self.feature_path(id) for id in feature_ids]
        self.video_ids = [id.split("/")[0] for id in feature_ids]
        self.target_paths = [self.VAE_feature_path(id) for id in self.video_ids]
        print("checking availability of all samples")
        existing_sample_paths = []
        not_found_count = 0
        for sample_path in tqdm(self.sample_paths):
            if os.path.exists(sample_path):
                existing_sample_paths.append(sample_path)
            else:
                not_found_count += 1

        self.sample_paths = existing_sample_paths
        print(f"{not_found_count} files were not found.")
        self.main_objects = main_objects
        self.sa_labels = sa_labels

    def __getitem__(self, index):
        sample_path = self.sample_paths[index]
        video_id = self.video_ids[index]
        target_path = self.target_paths[index]
        sample_type = self.feature_ids[index].split(".")[0][-1]
        # video_id = self.image_ids[index].split("/")[0]
        clip_features = np.load(sample_path)

        target_features = np.load(target_path)
        affordance_label = self.sa_labels[video_id]["affordance"]
        # affordance_sentense = self.args.affordance_sentences[affordance_label]
        multi_label_aff = self.sa_labels[video_id]["affordance_labels"]
        multi_label_aff[affordance_label] = 2
        object = self.sa_labels[video_id]["object"]

        return (
            clip_features,
            target_features,
            affordance_label,
            multi_label_aff,
            sample_type,
            sample_path,
            object,
        )

    def __len__(self):
        return len(self.sample_paths)

    def feature_path(self, feature_id):
        return os.path.join(self.args.image_featrures_dir, feature_id)

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

    train_dataset = ImageFeaturesDataset(train_ids, args, sa_labels, main_objects)
    val_dataset = ImageFeaturesDataset(val_ids, args, sa_labels, main_objects)
    test_dataset = ImageFeaturesDataset(test_ids, args, sa_labels, main_objects)

    subset_sampler = SubsetRandomSampler(train_dataset, subset_ratio=1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.AcE_batch_size,
        sampler=subset_sampler,
        # shuffle=True,
        num_workers=5,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.AcE_batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.AcE_batch_size,
        num_workers=5,
        shuffle=True,
    )

    return (
        train_dataset,
        train_loader,
        val_dataset,
        val_loader,
        test_dataset,
        test_loader,
    )
