import numpy as np
import cv2


def seg_to_bb(seg_mask, seg_value):
    segmentation = np.where(seg_mask == seg_value)
    if (
        len(segmentation) != 0
        and len(segmentation[1]) != 0
        and len(segmentation[0]) != 0
    ):
        x_min = int(np.min(segmentation[1]))
        x_max = int(np.max(segmentation[1]))
        y_min = int(np.min(segmentation[0]))
        y_max = int(np.max(segmentation[0]))

        return x_min, x_max, y_min, y_max
    else:
        return 0, 0, 0, 0


def image_wt_bb(img, bb):
    # height, width, _ = img.shape
    x_min, x_max, y_min, y_max = bb
    img_wt_bb = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 1)
    return img_wt_bb.astype(np.uint8)
