import torch
import torch.nn as nn
import clip


class AcE(nn.Module):
    def __init__(self, args):
        super(AcE, self).__init__()

        self.clip, _ = clip.load(args.CLIP_model, device=args.device)
        for param in self.clip.parameters():  # CLIP params are frozen
            param.requires_grad = False

        self.head = nn.Linear(self.clip.visual.output_dim, args.AcE_feature_size)

        for param in self.parameters():
            param.data = param.data.to(torch.float32)

    def forward(self, images):
        features = self.clip.encode_image(images)
        output = self.head(features)
        return output
