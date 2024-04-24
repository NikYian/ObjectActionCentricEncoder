from PIL import Image
import json
import os

from datasets.video_dataset import build_video_dataset
from models.teacher import load_teacher


def object_bb_from_annotations(args):
    """
    interacting_object_bb = {video_id: {'object': object_class
                                        'bounding_boxes': {frame:(xmin, ymin, xmax, ymax)
                                        }
                            }
    """
    interacting_object_bb = {}
    for root, dirs, files in os.walk(args.annotation_dir):
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


def generate_image_dataset(args):

    video_dataset, video_dataloader = build_video_dataset(args)
    video_dataset.get_whole_video_switch()
    iobb = object_bb_from_annotations(args)  # iobb = interacting object bounding boxes

    teacher = load_teacher(args)

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
    breakpoint()
