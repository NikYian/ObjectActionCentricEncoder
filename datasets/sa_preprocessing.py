import os
import json
from PIL import Image
from tqdm import tqdm


def frame2objcrop():
    jpg_dir = "/gpu-data2/nyian/ssv2/jpg"

    ann_path = "ssv2/somethings_affordances/annotations.json"

    output_dir = "/gpu-data2/nyian/ssv2/object_crops"

    with open(ann_path, "r") as f:
        box_annotations = json.load(f)

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
                        # pil_img = Image.fromarray(obj_crop)

                        output_path = os.path.join(output_dir, fname)
                        obj_crop.save(output_path)
                        break
            else:
                print(f"Frame image not found: {frame_path}")


if __name__ == "__main__":
    frame2objcrop()
