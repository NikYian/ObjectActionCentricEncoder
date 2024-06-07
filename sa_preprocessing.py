import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import clip

from models.teacher import load_teacher
from models.AcE import AcEnn
from datasets.video_dataset import build_video_dataset
from args import Args


def frame2objcrop(box_annotations):
    jpg_dir = "/gpu-data2/nyian/ssv2/jpg"
    output_dir = "/gpu-data2/nyian/ssv2/object_crops"

    for video_id, ann in tqdm(box_annotations.items()):
        directory_path = os.path.join(output_dir, video_id)
        os.makedirs(directory_path, exist_ok=True)
        annotations = ann["ann"]
        obj_id = ann["obj"]
        for frame in annotations:
            fname = frame["name"]
            frame_path = os.path.join(jpg_dir, fname)
            if os.path.exists(frame_path):

                image = Image.open(frame_path)

                for obj in frame["labels"]:
                    if obj["gt_annotation"] == obj_id:
                        bbox = obj["box2d"]
                        left = bbox["x1"]
                        upper = bbox["y1"]
                        right = bbox["x2"]
                        lower = bbox["y2"]
                        obj_crop = image.crop((left, upper, right, lower))
                        if obj_crop.size != (0, 0):
                            output_path = os.path.join(output_dir, fname)
                            obj_crop.save(output_path)
                        break
            else:
                print(f"Frame image not found: {frame_path}")


def generate_clip_features(args, box_annotations):
    CLIP, preprocess = clip.load(args.CLIP_model, device=args.device)
    crop_dir = "/gpu-data2/nyian/ssv2/object_crops"
    output_dir = "/gpu-data2/nyian/ssv2/CLIP_features"
    for video_id, ann in tqdm(box_annotations.items()):
        directory_path = os.path.join(output_dir, video_id)
        os.makedirs(directory_path, exist_ok=True)
        annotations = ann["ann"]
        affordance = ann["affordance"]
        aff_sentense = args.affordance_sentences[affordance]
        for frame in annotations:
            fname = frame["name"].split(".")[0]
            image_file_path = os.path.join(output_dir, fname + "_i.npy")
            text_file_path = os.path.join(output_dir, fname + "_t.npy")

            # Check if both files already exist
            if os.path.exists(image_file_path) and os.path.exists(text_file_path):
                continue  # Skip this frame if both files already exist

            crop_path = os.path.join(crop_dir, frame["name"])
            try:
                image = Image.open(crop_path)
            except IOError:
                print(f"Error opening image: {crop_path}")
                continue
            image_tensor = preprocess(image).to(args.device).unsqueeze(0)
            image_features = CLIP.encode_image(image_tensor)
            img_features_numpy = image_features.cpu().detach().numpy().reshape(512)
            np.save(image_file_path, img_features_numpy)

            tokenized_text = clip.tokenize([aff_sentense])
            tokenized_text = tokenized_text.to(args.device)
            text_features = CLIP.encode_text(tokenized_text)
            txt_features_numpy = text_features.cpu().detach().numpy().reshape(512)
            np.save(text_file_path, txt_features_numpy)


def generate_targets_from_teacer(args, video_ids):

    video_dataset, video_dataloader = build_video_dataset(args, video_ids)
    teacher = load_teacher(args)

    for video, video_id in tqdm(video_dataloader):
        video = video.to(args.device)
        features = teacher.forward_features(video)
        features_numpy = features.cpu().detach().numpy().reshape(384)
        file_path = os.path.join(args.VAE_features_dir, video_id[0])
        np.save(file_path + ".npy", features_numpy)


if __name__ == "__main__":
    args = Args()

    print("Importing annotations...")
    ann_path = "ssv2/somethings_affordances/annotations.json"
    with open(ann_path, "r") as f:
        box_annotations = json.load(f)
    video_ids = list(box_annotations.keys())

    # Add a terminal question
    response = input(
        "Do you want to proceed with extracting the interacting object crops? (yes/no): "
    )

    if response.lower() in ["yes", "y"]:
        frame2objcrop(box_annotations)

    response = input(
        "Do you want to proceed with extracting the representation targets using the teacher VideoMAE module? (yes/no): "
    )
    if response.lower() in ["yes", "y"]:
        generate_targets_from_teacer(args, video_ids)

    response = input(
        "Do you want to proceed with extracting the CLIP representations from the object crops? (yes/no): "
    )
    if response.lower() in ["yes", "y"]:
        generate_clip_features(args, box_annotations)

    print("Preprocessing completed")
