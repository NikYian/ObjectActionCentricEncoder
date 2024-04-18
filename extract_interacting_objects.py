import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

from externals.EgoHOS.mmsegmentation.mmseg.apis import (
    inference_segmentor,
    init_segmentor,
)
import mmcv
import glob
from tqdm import tqdm
import argparse
from PIL import Image
import numpy as np
from skimage.io import imsave
import warnings
import torch
from args import Args
import shutil
from externals.EgoHOS.mmsegmentation.visualize import visualize
from datasets.video_dataset import build_video_dataset

warnings.filterwarnings("ignore")
args = Args()

num_of_gpus = torch.cuda.device_count()
print(num_of_gpus)
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)

video_dataset, video_dataloader = build_video_dataset(args)
video_dataset.get_whole_video_switch()

device = torch.device("cuda")

models = []
# model that segments the two hands
models.append(
    init_segmentor(args.config_file_2h, args.checkpoint_file_2h, device=device)
)
# model that predicts the object/hand contact boundaries based on the output of the previous model
models.append(
    init_segmentor(args.config_file_cb, args.checkpoint_file_cb, device=device)
)
# model that segments the interacting object(s) of 1st and 2nd order
models.append(
    init_segmentor(args.config_file_obj2, args.checkpoint_file_obj2, device=device)
)

for dir in args.pred_seg_dir:
    shutil.rmtree(dir, ignore_errors=True)
    os.makedirs(dir)
shutil.rmtree(args.img_dir, ignore_errors=True)
os.makedirs(args.img_dir)

labels = [
    2,  # "Bending something so that it deforms"
    3,  # "Bending something until it breaks"
    5,  # "Closing something"
    14,  # "Folding something"
    22,  # "Letting something roll along a flat surface"
    134,  # "Something falling like a feather or paper"
    135,  # "Something falling like a rock"
    143,  # "Squeezing something"
    149,  # "Tearing something into two pieces"
    150,  # "Tearing something just a little bit"
    172,  # Unfolding something"
]

for video in video_dataset:
    if video[1] in labels:
        for i, image in enumerate(video[0]):
            imsave(
                os.path.join(args.img_dir, str(video[1]) + "_" + str(i) + ".jpg"),
                image.astype(np.uint8),
            )
        for file in tqdm(glob.glob(args.img_dir + "/*")):
            fname = os.path.basename(file).split(".")[0]
            img = np.array(Image.open(os.path.join(args.img_dir, fname + ".jpg")))
            for i in range(3):
                seg_result = inference_segmentor(models[i], file)[0]
                imsave(
                    os.path.join(args.pred_seg_dir[i], fname + ".png"),
                    seg_result.astype(np.uint8),
                )

        visualize(args)

        breakpoint()

    # seg_res = inference_segmentor(models[0], image)[0]
#  video_dataset[0][0].shape (35, 240, 320, 3)


# model = init_segmentor(args.config_file_obj1, args.checkpoint_file_obj1, device=device)

# for file in tqdm(glob.glob(args.img_dir + "/*")):
#     fname = os.path.basename(file).split(".")[0]
#     img = np.array(Image.open(os.path.join(args.img_dir, fname + ".jpg")))
#     seg_result = inference_segmentor(model, file)[0]
#     imsave(
#         os.path.join(args.pred_seg_dir[i], fname + ".png"),
#         seg_result.astype(np.uint8),
#     )
#     breakpoint()
