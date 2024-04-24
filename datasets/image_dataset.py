from PIL import Image
import json
import os
import clip
import torch
import numpy as np
import tqdm
from torchvision import transforms
from torch.utils.data import Dataset

from datasets.video_dataset import build_video_dataset
from models.teacher import load_teacher
import externals.VideoMAE.video_transforms as video_transforms
import externals.VideoMAE.volume_transforms as volume_transforms

"""
This implementation is based on
https://github.com/MCG-NJU/VideoMAE/blob/main/ssv2.py
pulished under CC-BY-NC 4.0 license: https://github.com/MCG-NJU/VideoMAE/blob/main/LICENSE
"""


class OAcEImgDataset(Dataset):
    def __init__(self, args):
        self.obj_crop_dir = args.obj_crop_dir
        self.VAE_features_dir = args.VAE_features_dir
        _, self.preprocess = clip.load(args.CLIP_model, device=args.device)

        cleaned_samples = np.array(cleaned.values[:, 0])
        cleaned_labels = np.array(cleaned.values[:, 1])

        self.data_transform = None

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
            return (
                buffer,
                self.label_array[index],
                sample.split("/")[-1].split(".")[0],
            )

    def __len__(self):
        return len(self.dataset_samples)


def object_bb_from_annotations(args):
    """
    interacting_object_bb = {video_id: {'object': object_class
                                        'bounding_boxes': {frame:(xmin, ymin, xmax, ymax)
                                        }
                            }
    """
    interacting_object_bb = {}
    for root, _, files in os.walk(args.annotation_dir):
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

    # generate object crops
    if gen_obj_crops:
        for video in video_dataset:
            video_id = video[2]
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
