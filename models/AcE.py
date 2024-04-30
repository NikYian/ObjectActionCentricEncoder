import torch
import torch.nn as nn
import clip
from models.teacher import load_teacher


class get_AcE(nn.Module):
    def __init__(self, args):
        super(get_AcE, self).__init__()

        self.clip, _ = clip.load(args.CLIP_model, device=args.device)
        for param in self.clip.parameters():  # CLIP params are frozen
            param.requires_grad = False

        self.head = nn.Linear(self.clip.visual.output_dim, args.AcE_feature_size)

        self.ac_head = load_teacher(args).head  # action classification head

        for param in self.parameters():
            param.data = param.data.to(torch.float32)

        if args.AcE_checkpoint:
            checkpoint = torch.load(args.AcE_checkpoint)
            self.head.load_state_dict(checkpoint)

    def forward(self, images):
        clip_features = self.clip.encode_image(images)
        features = self.head(clip_features)
        return features

    def predict_affordances(self, images):
        clip_features = self.clip.encode_image(images)
        features = self.head(clip_features)
        ssv2_label_logits = self.ac_head(features)
        res = torch.nn.functional.softmax(ssv2_label_logits, dim=-1)
        return res[:, 122], res[:, 144]
