import torch
import json
import numpy as np
import csv
from tqdm import tqdm
import pandas as pd
from scipy.special import softmax
from args import Args

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

# filename = "ssv2/something_else_ids.csv"

# # Open the file in write mode
# with open(filename, "w", newline="") as file:
#     # Loop through the list of IDs
#     for id in video_ids:
#         # Write each ID to a new line in the file
#         file.write(str(id) + "\n")

# action labels from Something's Affordance subset
labels_to_keep = np.array(list(args.action2aff_labels.keys()))

# affordances = set()
# # get affordance list
# for key, value in args.action2aff_labels.items():
#     affordances.add(value[2])
# affordances = np.array(list(affordances))

affordances = np.array(args.affordances)


# dataset info
somethings_aff_categories = {label: {"sample_num": 0} for label in labels_to_keep}
objects = {}
affordance_info = {}


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
                if args.action2aff_labels[label][1] == "object 0":
                    for frame in box_annotations[video_id]:
                        for item in frame["labels"]:
                            if item["gt_annotation"] == "object 0":
                                object = item["category"]
                                video_ann.append(frame)

                    # object = annotation["placeholders"][0]
                elif args.action2aff_labels[label][1] == "object 1":
                    for frame in box_annotations[video_id]:
                        for item in frame["labels"]:
                            if item["gt_annotation"] == "object 1":
                                object = item["category"]
                                video_ann.append(frame)
                    # object = annotation["placeholders"][0]

                if affordance in affordance_info:
                    affordance_info[affordance]["sample_num"] += 1
                    affordance_info[affordance]["objects"].add(object)
                else:
                    affordance_info[affordance] = {}
                    affordance_info[affordance]["sample_num"] = 1
                    affordance_info[affordance]["objects"] = {object}

                # generate object affordances from ssv2 actions and also count how many
                # times each object appears in the Something's Affordance subset
                if object in objects:
                    # objects[object]["affordance_labels"].add(label)
                    objects[object]["video_ids"].append(video_id)
                    objects[object]["sample_num"] += 1
                    objects[object]["affordances"].add(args.action2aff_labels[label][2])
                    objects[object]["affordance_distribution"][
                        np.where(affordances == affordance)[0][0]
                    ] += 1
                else:
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

                sa_ann[video_id] = {
                    "ann": video_ann,
                    "obj": args.action2aff_labels[label][1],
                }

                # count how many samples in each affordance category
                somethings_aff_categories[label]["sample_num"] += 1
                ssv2_annotations[video_id] = annotation
                ssv2_annotations[video_id]["label"] = label

# with open("ssv2/somethings_affordances/annotations.json", "w") as json_file:
#     json.dump(sa_ann, json_file, indent=4)

main_objects = {}
for object in objects:
    if objects[object]["sample_num"] > 50:
        mask = objects[object]["affordance_distribution"] > 30
        main_objects[object] = objects[object]
        main_objects[object]["affordance_labels"] = [
            int(item) for item in np.where(mask, 1, 0)
        ]
        # main_objects[object]["video_ids"] = main_objects[object]["video_ids"].to_list()
        main_objects[object]["affordance_distribution"] = main_objects[object][
            "affordance_distribution"
        ].tolist()
        main_objects[object]["affordances"] = list(main_objects[object]["affordances"])


with open("ssv2/somethings_affordances/main_objects.json", "w") as json_file:
    json.dump(main_objects, json_file, indent=4)
    # sum = np.sum(objects[object]["affordance_distribution"])
    # objects[object]["affordance_distribution"] = list(
    #     objects[object]["affordance_distribution"] / sum
    # )

train_ids = []
val_ids = []
test_ids = []


## create train,test split from video ids

for object in objects:
    if object in main_objects:
        objects[object]["video_ids"] = np.array(objects[object]["video_ids"])
        np.random.shuffle(objects[object]["video_ids"])

        split1 = int(len(objects[object]["video_ids"]) * args.split_ratios[0])
        split2 = split1 + int(len(objects[object]["video_ids"]) * args.split_ratios[1])

        train_ids.extend(objects[object]["video_ids"][:split1])
        val_ids.extend(objects[object]["video_ids"][split1:split2])
        test_ids.extend(objects[object]["video_ids"][split2:])
    else:
        train_ids.extend(objects[object]["video_ids"])
breakpoint()
with open("ssv2/somethings_affordances/train.json", "w") as json_file:
    json.dump(train_ids, json_file)
with open("ssv2/somethings_affordances/val.json", "w") as json_file:
    json.dump(val_ids, json_file)
with open("ssv2/somethings_affordances/test.json", "w") as json_file:
    json.dump(test_ids, json_file)


# objects_df = pd.DataFrame.from_dict(main_objects, orient="index")
# objects_df = objects_df.sort_values(by="sample_num", ascending=False)
# objects_dict = objects_df.to_dict(orient="index")
# objects_json = pd.Series(objects_dict).to_json(orient="records")

# with open("ssv2/somethings_affordances/main_objects.json", "w") as f:
#     f.write(objects_json)
# objects_df2 = objects_df[["affordance_labels", "sample_num"]]
# objects_df2.to_csv("ssv2/somethings_affordances/sa_objects.csv", index=True)

# video_ids = list(sa_ann.keys())
# filename = "ssv2/somethings_aff_ids.csv"
# # Open the file in write mode
# with open(filename, "w", newline="") as file:
#     # Loop through the list of IDs
#     for id in video_ids:
#         # Write each ID to a new line in the file
#         file.write(str(id) + "\n")

# somethings_aff_categories = pd.DataFrame.from_dict(
#     somethings_aff_categories, orient="index"
# )
# affordance_info_df = pd.DataFrame.from_dict(affordance_info, orient="index")
print("Annotation extraction completed")
