import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
from utils import seg_to_bb, image_wt_bb

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

# object_labels = [3, 4, 5]


for indx, video in enumerate(video_dataloader):
    for dir in args.pred_seg_dir:
        shutil.rmtree(dir, ignore_errors=True)
        os.makedirs(dir)
    shutil.rmtree(args.img_dir, ignore_errors=True)
    os.makedirs(args.img_dir)
    shutil.rmtree(args.cb_view, ignore_errors=True)
    os.makedirs(args.cb_view)
    label = video[1][0].item()
    for i, image in enumerate(video[0]):
        image = image[0].numpy()
        imsave(
            os.path.join(args.img_dir, str(label) + "_" + str(i) + ".jpg"),
            image.astype(np.uint8),
        )
    video_len = len(video[0])
    cb_densities = [0] * video_len
    obj_densities = [0] * video_len
    obj_densities_labels = [0] * video_len

    for file in tqdm(glob.glob(args.img_dir + "/*")):
        fname = os.path.basename(file).split(".")[0]
        img = np.array(Image.open(os.path.join(args.img_dir, fname + ".jpg")))
        frame_num = int(fname.split("_")[1])
        for i in range(3):
            seg_result = inference_segmentor(models[i], file)[0]
            if i == 1:  # model that predicts the contact boundary
                cb_density = np.count_nonzero(seg_result == 1)
                cb_densities[frame_num] = cb_density
                imsave(
                    args.cb_view + "/" + fname + "_" + str(cb_density) + ".png",
                    (seg_result * 255).astype(np.uint8),
                )

            imsave(
                os.path.join(args.pred_seg_dir[i], fname + ".png"),
                seg_result.astype(np.uint8),
            )
            if i == 2:
                seg_labels = np.unique(seg_result)
                for seg_value in seg_labels:
                    if seg_value != 0:
                        obj_density = np.count_nonzero(seg_result == seg_value)
                        obj_densities_labels[frame_num] = seg_value
                        obj_densities[frame_num] = max(
                            obj_densities[frame_num], obj_density
                        )

    # closest to the mean cb density
    # mean_density = np.mean(cb_densities)
    # cb_densities = np.abs(cb_densities - mean_density)
    # frame_num = np.argmin(cb_densities)

    #  max cb density
    #  biggest_cb_index = np.argmax(cb_densities)

    # closest to the mean obj density
    # mean_density = np.mean(obj_densities)
    # obj_densities = np.abs(obj_densities - mean_density)
    # frame_num = np.argmin(obj_densities)

    # median obj density
    sorted_indices = np.argsort(obj_densities)
    median_index = len(obj_densities) // 2
    frame_num = sorted_indices[median_index]
    if obj_densities[frame_num] == 0:
        # closest to the mean obj density
        mean_density = np.mean(obj_densities)
        obj_densities = np.abs(obj_densities - mean_density)
        frame_num = np.argmin(obj_densities)

    print(f"getting object at frame {frame_num}")
    file = args.img_dir + "/" + str(label) + "_" + str(frame_num) + ".jpg"
    seg_result = inference_segmentor(models[2], file)[0]
    seg_labels = np.unique(seg_result)

    img = np.array(Image.open(file))
    obj_bb = seg_to_bb(seg_result, obj_densities_labels[frame_num])
    imsave(
        os.path.join(
            args.tests_dir,
            str(indx) + "_" + str(label) + "_" + video[2][0] + "_bb.png",
        ),
        image_wt_bb(img, obj_bb),
    )

    # visualize(args)
