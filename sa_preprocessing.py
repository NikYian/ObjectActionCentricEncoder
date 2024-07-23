import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import json
from PIL import Image
from tqdm import tqdm
import numpy as np
import clip
import sys
import torchvision.transforms as transforms

from models.teacher import load_teacher
from models.AcE import AcEnn
from datasets.video_dataset import build_video_dataset
from args import Args
import torchvision.datasets as datasets

sys.path.append(r"./externals/mae")

import models_mae

sys.path.append(r"./externals")
from SOLV.models.model import SOLV_nn, Visual_Encoder, MLP
from SOLV.read_args import get_args, print_args

import utils.SOLV_utils as utils


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


def prepare_model(chkpt_dir, arch="mae_vit_base_patch16"):
    # build model
    model = getattr(models_mae, arch)()
    # load model
    checkpoint = torch.load(chkpt_dir, map_location="cpu")
    msg = model.load_state_dict(checkpoint["model"], strict=False)
    print(msg)
    return model


def generate_mae_features(args, box_annotations):
    chkpt_dir = "/gpu-data2/nyian/checkpoints/mae_vit_base_patch16.pth"
    model_mae = prepare_model(chkpt_dir, "mae_vit_base_patch16").to(args.device)
    crop_dir = "/gpu-data2/nyian/ssv2/object_crops"
    dataset = datasets.ImageFolder(crop_dir)
    output_dir = "/gpu-data2/nyian/ssv2/mae_features"
    for video_id, ann in tqdm(box_annotations.items()):
        directory_path = os.path.join(output_dir, video_id)
        os.makedirs(directory_path, exist_ok=True)
        annotations = ann["ann"]
        affordance = ann["affordance"]
        aff_sentense = args.affordance_sentences[affordance]
        for frame in annotations:
            fname = frame["name"].split(".")[0]
            feature_path = os.path.join(output_dir, fname + ".npy")

            # Check if both files already exist
            if os.path.exists(feature_path):
                continue  # Skip this frame if file already exist

            crop_path = os.path.join(crop_dir, frame["name"])
            try:
                image = Image.open(crop_path)
            except IOError:
                print(f"Error opening image: {crop_path}")
                continue

            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),  # Resize to 224x224
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),  # Normalization with ImageNet mean and std
                ]
            )
            image = dataset.loader(crop_path)
            image_tensor = transform(image).unsqueeze(0).to(args.device)
            image_features = model_mae.forward_encoder(image_tensor, 0)
            image_features = model_mae.norm(image_features[0].mean(1))
            img_features_numpy = image_features.cpu().detach().numpy().reshape(768)
            np.save(feature_path, img_features_numpy)


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
            image_file_path = os.path.join(output_dir, fname + ".npy")

            # Check if both files already exist
            if os.path.exists(image_file_path):
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

            # tokenized_text = clip.tokenize([aff_sentense])
            # tokenized_text = tokenized_text.to(args.device)
            # text_features = CLIP.encode_text(tokenized_text)
            # txt_features_numpy = text_features.cpu().detach().numpy().reshape(512)
            # np.save(text_file_path, txt_features_numpy)


def generate_targets_from_teacer(args, video_ids):

    video_dataset, video_dataloader = build_video_dataset(args, video_ids)
    teacher = load_teacher(args)

    for video, video_id in tqdm(video_dataloader):
        video = video.to(args.device)
        features = teacher.forward_features(video)
        features_numpy = features.cpu().detach().numpy().reshape(384)
        file_path = os.path.join(args.VAE_features_dir, video_id[0])
        np.save(file_path + ".npy", features_numpy)


@torch.no_grad()
def generate_SOLV_features_for_AcE(SOLV_args):
    train_dataloader, val_dataloader, test_loader = utils.get_dataloaders(SOLV_args)
    loaders = [train_dataloader, val_dataloader, test_loader]
    # train_ids = []
    # val_ids = []
    # test_ids = []
    # ids = [train_ids, val_ids, test_ids]
    splits = ["train", "val", "test"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vis_encoder = Visual_Encoder(SOLV_args).to(device)
    SOLV_model = SOLV_nn(SOLV_args).to(device)

    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        SOLV_args.checkpoint_path,
        run_variables=to_restore,
        model=SOLV_model,
        remove_module_from_key=True,
    )
    vis_encoder.eval()
    SOLV_model.eval()

    inputs_dir = "/gpu-data2/nyian/ssv2/SOLV/inputs"
    targets_dir = "/gpu-data2/nyian/ssv2/SOLV/targets"

    for j, loader in enumerate(loaders):
        for (
            frames,
            input_masks,
            video_id,
            frame_idx,
            multi_label_aff,
            bb_masks,
            target_frame_paths,
        ) in tqdm(loader):

            frames = frames.cuda()  # (1, #frames + 2N, 3, H, W)
            input_masks = input_masks.cuda()  # (1, #frames + 2N)

            output_features, token_indices = vis_encoder(frames, get_gt=False)

            output_features = output_features.to(torch.float32).cuda()

            reconstruction = SOLV_model(output_features, input_masks, token_indices)

            output_features = output_features.to(torch.float32)
            inputs = reconstruction["target_frame_slots"].cpu().detach().numpy()
            targets = reconstruction["slots_temp"].cpu().detach().numpy()
            for i in range(inputs.shape[0]):
                path = os.path.join(
                    inputs_dir,
                    splits[j],
                    video_id[i] + "_" + str(frame_idx[i].item()) + ".npy",
                )
                if os.path.exists(path):
                    continue
                np.save(path, inputs[i])
                path = os.path.join(
                    targets_dir,
                    splits[j],
                    video_id[i] + "_" + str(frame_idx[i].item()) + ".npy",
                )
                np.save(path, targets[i])


if __name__ == "__main__":
    SOLV_args = get_args()
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
        print("CLIP features extracted succesfully")

    response = input(
        "Do you want to proceed with extracting the MAE representations from the object crops? (yes/no): "
    )
    if response.lower() in ["yes", "y"]:
        generate_mae_features(args, box_annotations)
        print("MAE features extracted succesfully")

    response = input(
        "Do you want to proceed with extracting the SOLV representations? (yes/no): "
    )
    if response.lower() in ["yes", "y"]:
        generate_SOLV_features_for_AcE(SOLV_args)
        print("SOLV features extracted succesfully")

    print("Preprocessing completed")
