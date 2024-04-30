from PIL import Image
import json
import os
import clip
import torch
import numpy as np
import tqdm
from torchvision import transforms
from torch.utils.data import Dataset, SubsetRandomSampler, DataLoader

from datasets.video_dataset import build_video_dataset
from models.teacher import load_teacher
import externals.VideoMAE.video_transforms as video_transforms
import externals.VideoMAE.volume_transforms as volume_transforms


class OAcEImgDataset(Dataset):
    def __init__(self, args):
        # self.obj_crop_dir = args.obj_crop_dir
        # self.VAE_features_dir = args.VAE_features_dir
        _, self.preprocess = clip.load(args.CLIP_model, device=args.device)

        self.sample_paths = [
            os.path.join(args.obj_crop_dir, file)
            for file in os.listdir(args.obj_crop_dir)
        ]
        self.label_paths = {
            fname.split(".")[0]: os.path.join(args.VAE_features_dir, fname)
            for fname in os.listdir(args.VAE_features_dir)
        }
        self.data_transform = None

    def __getitem__(self, index):
        sample = Image.open(self.sample_paths[index])
        sample = self.preprocess(sample).to(torch.float32)
        fname = os.path.basename(self.sample_paths[index])
        video_id = fname.split("_")[0]
        object = fname.split("_")[1]
        if object == "object" or object == "Object":
            object_id = 0
        elif object == "ball":
            object_id = 1
        else:
            object_id = 2
        label = np.load(self.label_paths[video_id])

        return sample, label, object_id, video_id

    def __len__(self):
        return len(self.sample_paths)


def object_bb_from_annotations(args):
    """
    interacting_object_bb = {video_id: {'object': object_class
                                        'bounding_boxes': {frame:(xmin, ymin, xmax, ymax)
                                        }
                            }
    """
    interacting_object_bb = {}
    for root, dir, files in os.walk(args.annotation_dir):
        for filename in files:
            filepath = os.path.join(root, filename)
            video_id = filename.split(".")[0]
            with open(filepath, "r") as f:
                data = json.load(f)
            obj_bbs = {}
            for frame in data["frames"]:
                bb = frame["figures"][0]["geometry"]["points"]["exterior"]
                top_left = bb[0]
                bottom_right = bb[1]
                xmin, ymin = top_left
                xmax, ymax = bottom_right
                obj_bbs[frame["index"]] = (xmin, ymin, xmax, ymax)
            interacting_object_bb[video_id] = {}
            interacting_object_bb[video_id]["bounding_boxes"] = obj_bbs
            interacting_object_bb[video_id]["object"] = data["objects"][0]["classTitle"]

    return interacting_object_bb


def generate_image_dataset(args, gen_obj_crops=True, gen_VAE_features=True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    video_dataset, video_dataloader = build_video_dataset(args)
    video_dataset.get_whole_video_switch()
    iobb = object_bb_from_annotations(args)  # iobb = interacting object bounding boxes

    ids = []
    for video in video_dataset:
        ids.append(video[2])

    breakpoint()
    # generate object crops
    if gen_obj_crops:
        for video in video_dataset:
            video_id = video[2]
            print(video_id)
            if video_id in iobb:
                bbs = iobb[video_id]["bounding_boxes"]
                object = iobb[video_id]["object"]

                frames = video[0]
                for frame, bb in bbs.items():
                    xmin, ymin, xmax, ymax = bb
                    obj_crop = frames[frame][ymin:ymax, xmin:xmax]
                    pil_img = Image.fromarray(obj_crop)
                    pil_img.save(
                        args.obj_crop_dir
                        + "/"
                        + str(video_id)
                        + "_"
                        + object
                        + "_"
                        + str(frame)
                        + ".jpg"
                    )

    # generate VAE_features
    if gen_VAE_features:
        teacher = load_teacher(args)

        video_dataset.get_whole_video_switch()

        for video in tqdm.tqdm(video_dataloader):
            video_id = video[2][0]
            video = video[0].to(device)
            features = teacher.forward_features(video)
            features_numpy = features.cpu().detach().numpy().reshape(384)
            file_path = os.path.join(args.VAE_features_dir, video_id)
            np.save(file_path + ".npy", features_numpy)

    # create torch Dataset and Dataloader classes
    dataset = OAcEImgDataset(args)

    torch.manual_seed(42)
    indices = torch.randperm(len(dataset))

    train_ratio = args.split_ratios[0]
    val_ratio = args.split_ratios[1]

    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))

    train_indices = indices[:train_size]
    val_indices = indices[train_size : train_size + val_size]
    test_indices = indices[train_size + val_size :]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset, sampler=train_sampler, batch_size=args.AcE_batch_size
    )
    val_loader = DataLoader(
        dataset, sampler=val_sampler, batch_size=args.AcE_batch_size
    )
    test_loader = DataLoader(
        dataset, sampler=test_sampler, batch_size=args.AcE_batch_size
    )

    return dataset, train_loader, val_loader, test_loader
