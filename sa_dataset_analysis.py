import torch
import json
import numpy as np
import csv
import random
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
from args import Args
from glob import glob
import os


def check_bb(left, upper, right, lower):
    width = int(right - left)
    height = int(lower - upper)

    if width <= 0 or height <= 0:
        return False
    else:
        return True


args = Args()

print("... Loading box annotations might take a minute ...")
box_annotations = {}

for ann_path in args.something_else_ann:
    with open(ann_path, "r") as f:
        annotations = json.load(f)
        box_annotations.update(annotations)
print("Loading box annotations successful.")
sa_ann = {}

# video ids of all videos annotated at Something-Else
video_ids = list(box_annotations.keys())

# action labels from Something's Affordance subset
labels_to_keep = np.array(list(args.action2aff_labels.keys()))

affordances = np.array(args.affordances)

# dataset info
somethings_aff_categories = {label: {"sample_num": 0} for label in labels_to_keep}
objects = {}
affordance_info = {}
sa_labels = {}

with open("ssv2/labels.json", "r") as f:
    label_encoder = json.load(f)

ssv2_ann_dirs = ["ssv2/train.json", "ssv2/validation.json"]
ssv2_annotations = {}

for dirs in ssv2_ann_dirs:
    with open(dirs, "r") as f:
        annotations = json.load(f)
        for annotation in tqdm(annotations, desc="Processing annotations"):
            template = annotation["template"]
            template = template.replace("[", "").replace("]", "")
            label = int(label_encoder[template])

            video_id = annotation["id"]

            if label in labels_to_keep and video_id in video_ids:
                affordance = args.action2aff_labels[label][2]

                video_ann = []
                bbs = []
                good_frames = []  # frames that have a bb of the interacting object
                if args.action2aff_labels[label][1] == "object 0":
                    for frame in box_annotations[video_id]:
                        for item in frame["labels"]:
                            if item["gt_annotation"] == "object 0":
                                object = item["category"]
                                bbox = item["box2d"]
                                left = bbox["x1"]
                                upper = bbox["y1"]
                                right = bbox["x2"]
                                lower = bbox["y2"]
                                bb = (left, upper, right, lower)
                                if check_bb(left, upper, right, lower):
                                    video_ann.append(frame)
                                    bbs.append(bb)
                                    good_frames.append(frame["name"])

                    # object = annotation["placeholders"][0]
                elif args.action2aff_labels[label][1] == "object 1":
                    for frame in box_annotations[video_id]:
                        for item in frame["labels"]:
                            if item["gt_annotation"] == "object 1":
                                object = item["category"]
                                video_ann.append(frame)
                                bbox = item["box2d"]
                                left = bbox["x1"]
                                upper = bbox["y1"]
                                right = bbox["x2"]
                                lower = bbox["y2"]
                                bb = (left, upper, right, lower)
                                if check_bb(left, upper, right, lower):
                                    video_ann.append(frame)
                                    bbs.append(bb)
                                    good_frames.append(frame["name"])

                    # object = annotation["placeholders"][0]

                if affordance in affordance_info and video_ann:
                    affordance_info[affordance]["sample_num"] += 1
                    affordance_info[affordance]["objects"].add(object)
                elif video_ann:
                    affordance_info[affordance] = {}
                    affordance_info[affordance]["sample_num"] = 1
                    affordance_info[affordance]["objects"] = {object}

                # generate object affordances from ssv2 actions and also count how many
                # times each object appears in the Something's Affordance subset
                if object in objects and video_ann:
                    # objects[object]["affordance_labels"].add(label)
                    objects[object]["video_ids"].append(video_id)
                    objects[object]["sample_num"] += 1
                    objects[object]["affordances"].add(args.action2aff_labels[label][2])
                    objects[object]["affordance_distribution"][
                        np.where(affordances == affordance)[0][0]
                    ] += 1
                elif video_ann:
                    objects[object] = {}
                    objects[object]["video_ids"] = [video_id]
                    objects[object]["affordance_distribution"] = np.zeros(
                        len(affordances)
                    )
                    objects[object]["affordance_distribution"][
                        np.where(affordances == affordance)[0][0]
                    ] += 1
                    # objects[object]["affordance_labels"] = {label}
                    objects[object]["sample_num"] = 1
                    objects[object]["affordances"] = {args.action2aff_labels[label][2]}

                if video_ann:
                    sa_ann[video_id] = {
                        "ann": video_ann,
                        "obj": args.action2aff_labels[label][1],
                        "object": object,
                        "affordance": int(np.where(affordances == affordance)[0][0]),
                        "bbs": bbs,
                        "good_frames": good_frames,
                    }

                # count how many samples in each affordance category
                somethings_aff_categories[label]["sample_num"] += 1
                ssv2_annotations[video_id] = annotation
                ssv2_annotations[video_id]["label"] = label


with open("ssv2/somethings_affordances/annotations.json", "w") as json_file:
    json.dump(sa_ann, json_file)

main_objects = {}
for object in objects:
    if objects[object]["sample_num"] > 20:
        sample_num = objects[object]["sample_num"]
        mask1 = objects[object]["affordance_distribution"] / sample_num > 0.2
        mask2 = objects[object]["affordance_distribution"] > 50
        mask = mask1 | mask2
        main_objects[object] = objects[object]
        main_objects[object]["affordance_labels"] = [
            int(item) for item in np.where(mask, 1, 0)
        ]
        # main_objects[object]["video_ids"] = main_objects[object]["video_ids"].to_list()
        main_objects[object]["affordance_distribution"] = main_objects[object][
            "affordance_distribution"
        ].tolist()
        main_objects[object]["affordances"] = list(main_objects[object]["affordances"])

main_objects_list = list(main_objects.keys())
random.seed(3)
random.shuffle(main_objects_list)
split_index = int(0.6 * len(main_objects_list))
setA = main_objects_list[:split_index]
setB = main_objects_list[split_index:]

main_objects_df = pd.DataFrame.from_dict(main_objects, orient="index")
main_objects_df = main_objects_df.sort_values(by="sample_num", ascending=False)


with open("ssv2/somethings_affordances/main_objects.json", "w") as json_file:
    json.dump(main_objects, json_file, indent=4)

main_objects_set = set(main_objects.keys())
for affordance in affordance_info.keys():
    negative_objects = []
    for object in main_objects.keys():
        if object not in affordance_info[affordance]["objects"]:
            negative_objects.append(object)
    aff_objects = main_objects_set & affordance_info[affordance]["objects"]
    affordance_info[affordance]["objects"] = aff_objects
    affordance_info[affordance]["negative_objects"] = negative_objects


train_ids = []
val_ids = []
test_ids = []

train_video_ids = []
val_video_ids = []
test_video_ids = []

## create train,test split from video ids

for object in objects:
    if object in main_objects:
        objects[object]["video_ids"] = np.array(objects[object]["video_ids"])
        np.random.shuffle(objects[object]["video_ids"])

        split1 = int(len(objects[object]["video_ids"]) * args.split_ratios[0])
        split2 = split1 + int(len(objects[object]["video_ids"]) * args.split_ratios[1])

        train_video_ids.extend(objects[object]["video_ids"][:split1])
        val_video_ids.extend(objects[object]["video_ids"][split1:split2])
        test_video_ids.extend(objects[object]["video_ids"][split2:])
    else:
        train_video_ids.extend(objects[object]["video_ids"])

# create a list with all the frames of the videos that contain detected object. ex. "151201/0001.jpg"
video_id_lists = [test_video_ids, train_video_ids, val_video_ids]
frame_id_lists = [test_ids, train_ids, val_ids]

train_videos = []
test_videos = []
val_videos = []

video_lists = [test_videos, train_videos, val_videos]

for i in range(3):
    video_id_list = video_id_lists[i]
    frame_list = frame_id_lists[i]
    video_list = video_lists[i]
    for video_id in video_id_list:
        if video_id in sa_ann:
            fname = "/gpu-data2/nyian/ssv2/jpg/" + video_id
            img_list = sorted(
                glob(os.path.join(fname, "*.jpg")),
                key=lambda x: int(x.split("/")[-1].split(".")[0]),
            )
            frame_num = len(img_list)
            if frame_num == 0:
                print(f"no frames in {fname}")
            else:
                video_list.append({"id": video_id, "length": frame_num})
            for index, frame in enumerate(sa_ann[video_id]["ann"]):
                fname = frame["name"].split(".")[0]
                if i == 1:
                    frame_list.append(fname + ".npy")
                    # frame_list.append(fname + "_t.npy")
                else:
                    frame_list.append(fname + ".npy")
                if (
                    index == 10
                ):  # keep only frames from beggining of video to reduce object interference
                    break

with open("ssv2/somethings_affordances/train.json", "w") as json_file:
    json.dump(train_ids, json_file)
with open("ssv2/somethings_affordances/val.json", "w") as json_file:
    json.dump(val_ids, json_file)
with open("ssv2/somethings_affordances/test.json", "w") as json_file:
    json.dump(test_ids, json_file)

with open("ssv2/somethings_affordances/train_videos_.json", "w") as json_file:
    json.dump(train_videos, json_file)
with open("ssv2/somethings_affordances/val_videos_.json", "w") as json_file:
    json.dump(val_videos, json_file)
with open("ssv2/somethings_affordances/test_videos_.json", "w") as json_file:
    json.dump(test_videos, json_file)


## create compostional train,test split from video ids

train_ids = []
val_ids = []
test_ids = []

train_video_ids = []
val_video_ids = []
test_video_ids = []

for object in objects:
    # if (object in main_objects and object in setA) or object not in main_objects:
    if object in main_objects and object in setA:
        objects[object]["video_ids"] = np.array(objects[object]["video_ids"])
        train_video_ids.extend(objects[object]["video_ids"])
    else:
        objects[object]["video_ids"] = np.array(objects[object]["video_ids"])
        np.random.shuffle(objects[object]["video_ids"])
        split = int(len(objects[object]["video_ids"]) * 0.5)
        test_video_ids.extend(objects[object]["video_ids"][:split])
        val_video_ids.extend(objects[object]["video_ids"][split:])


# create a list with all the frames of the videos that contain detected object. ex. "151201/0001.jpg"
video_id_lists = [test_video_ids, train_video_ids, val_video_ids]
frame_id_lists = [test_ids, train_ids, val_ids]
# train_no_text = []

train_videos = []
test_videos = []
val_videos = []

video_lists = [test_videos, train_videos, val_videos]

for i in range(3):
    video_id_list = video_id_lists[i]
    frame_list = frame_id_lists[i]
    video_list = video_lists[i]
    for video_id in video_id_list:
        if video_id in sa_ann:
            fname = "/gpu-data2/nyian/ssv2/jpg/" + video_id
            img_list = sorted(
                glob(os.path.join(fname, "*.jpg")),
                key=lambda x: int(x.split("/")[-1].split(".")[0]),
            )
            frame_num = len(img_list)
            if frame_num == 0:
                print(f"no frames in {fname}")
            else:
                video_list.append({"id": video_id, "length": frame_num})
            for index, frame in enumerate(sa_ann[video_id]["ann"]):
                fname = frame["name"].split(".")[0]
                if i == 1:
                    # frame_list.append(fname + "_i.npy")
                    # frame_list.append(fname + "_t.npy")
                    frame_list.append(fname + ".npy")
                else:
                    frame_list.append(fname + ".npy")
                if (
                    index == 10
                ):  # keep only frames from beggining of video to reduce object interference
                    break


with open("ssv2/somethings_affordances/train_comp.json", "w") as json_file:
    json.dump(train_ids, json_file)
with open("ssv2/somethings_affordances/val_comp.json", "w") as json_file:
    json.dump(val_ids, json_file)
with open("ssv2/somethings_affordances/test_comp.json", "w") as json_file:
    json.dump(test_ids, json_file)
# with open("ssv2/somethings_affordances/train_comp_no_text.json", "w") as json_file:
#     json.dump(train_no_text, json_file)

with open("ssv2/somethings_affordances/train_videos.json", "w") as json_file:
    json.dump(train_videos, json_file)
with open("ssv2/somethings_affordances/val_videos.json", "w") as json_file:
    json.dump(val_videos, json_file)
with open("ssv2/somethings_affordances/test_videos.json", "w") as json_file:
    json.dump(test_videos, json_file)


for video_id in sa_ann.keys():
    sa_labels[video_id] = {
        "bbs": sa_ann[video_id]["bbs"],
        "good_frames": sa_ann[video_id]["good_frames"],
    }
    for frame in sa_ann[video_id]["ann"]:

        object = sa_labels[video_id]["object"] = sa_ann[video_id]["object"]
        sa_labels[video_id]["affordance"] = int(sa_ann[video_id]["affordance"])
        if object in main_objects:
            sa_labels[video_id]["affordance_labels"] = main_objects[object][
                "affordance_labels"
            ]
        else:
            sa_labels[video_id]["affordance_labels"] = [0] * len(affordances)
            sa_labels[video_id]["affordance_labels"][sa_ann[video_id]["affordance"]] = 1

with open("ssv2/somethings_affordances/sa_labels.json", "w") as json_file:
    json.dump(sa_labels, json_file)


# for object in main_objects:
#     video_ids =


# choose from N test set
def get_random_sample(object):
    video_ids = main_objects[object]["video_ids"]
    random_video_id = random.choice(video_ids[:10])
    frames = sa_ann[random_video_id]["ann"]
    if frames:
        random_frame = random.choice(frames)
    else:
        breakpoint()
    fname = random_frame["name"].split(".")[0]
    return fname + ".npy"


choose_from_N = []
for affordance, aff_info in affordance_info.items():
    positive_objects_set = aff_info["objects"] & set(setB)
    negative_objects_set = set(aff_info["negative_objects"]) & set(setB)
    for i in range(100):
        positive_object = random.sample(positive_objects_set, 1)[0]
        samples = [get_random_sample(positive_object)]
        negative_objects = random.sample(negative_objects_set, 4)
        negative_samples = [
            get_random_sample(negative_object) for negative_object in negative_objects
        ]
        label = int(np.where(affordances == affordance)[0][0])
        samples.extend(negative_samples)
        choose_from_N.append({"samples": samples, "label": label})


with open("ssv2/somethings_affordances/choose_from_N.json", "w") as json_file:
    json.dump(choose_from_N, json_file)


print("Annotation extraction completed")
