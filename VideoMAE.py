import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


from transformers import VideoMAEForPreTraining, VideoMAEForVideoClassification
import numpy as np
import torch

video = list(np.random.randn(16, 3, 224, 224))
VideoMAE = VideoMAEForPreTraining.from_pretrained(
    "MCG-NJU/videomae-base-short-finetuned-ssv2"
)
# model = VideoMAEForVideoClassification.from_pretrained(
#     "MCG-NJU/videomae-base-short-finetuned-ssv2"
# )

breakpoint()
