import torch
import torch.nn as nn
import clip
from models.teacher import load_teacher
from torch.nn.functional import softmax


class AcEnn(nn.Module):
    def __init__(self, args):
        super(AcEnn, self).__init__()
        self.args = args

        self.clip, self.preprocess = clip.load(args.CLIP_model, device=args.device)
        for param in self.clip.parameters():  # CLIP params are frozen
            param.requires_grad = False
        # for param in self.preprocess.parameters():
        #     param.requires_grad = False

        self.head = nn.Linear(self.clip.visual.output_dim, args.AcE_feature_size)

        self.aff_anchors = None

        # self.ac_head = load_teacher(args).head  # action classification head

        for param in self.parameters():
            param.data = param.data.to(torch.float32)

        if args.AcE_checkpoint:
            checkpoint = torch.load(args.AcE_checkpoint)
            self.head.load_state_dict(checkpoint)

    def forward(self, images):
        # images = self.preprocess(images)
        clip_features = self.clip.encode_image(images)
        features = self.head(clip_features)
        return features

    def forward_text(self, texts):
        tokenized_text = clip.tokenize(texts)
        tokenized_text = tokenized_text.to(self.args.device)
        clip_features = self.clip.encode_text(tokenized_text)
        features = self.head(clip_features)
        return features

    def update_aff_anchors(self):
        self.aff_anchors = []
        for aff_sentense in self.args.affordance_sentences:
            anchor_features = self.forward_text(aff_sentense)
            self.aff_anchors.append(anchor_features)
        self.aff_anchors = torch.cat(self.aff_anchors, dim=0)

    def ZS_predict(self, images):

        image_features = self.forward(images)
        features_norm = torch.nn.functional.normalize(image_features, dim=1)
        anchors_norm = torch.nn.functional.normalize(self.aff_anchors, dim=1)
        similarities = torch.mm(features_norm, torch.transpose(anchors_norm, 0, 1))
        similarities = similarities + 1
        similarities = similarities / 2
        return similarities

    # def predict_affordances(self, images):
    #     images = self.preprocess(images)
    #     clip_features = self.clip.encode_image(images)
    #     batch_size = images.shape[0]
    #     features = self.head(clip_features)
    #     affordance_logits = self.ac_head(features).detach().cpu().numpy()
    #     # apply softmax in pairs of 2
    #     reshaped_tensor = torch.tensor(affordance_logits).view(batch_size, 7, 2)
    #     softmaxed_pairs = softmax(reshaped_tensor, dim=2)
    #     output = softmaxed_pairs.view(affordance_logits.shape).numpy()
    #     aff = output[:, self.args.affordance_indices]
    #     return aff
